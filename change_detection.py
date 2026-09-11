import geopandas as gpd
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


def _write_change_detection_geojson(
    joined_gdf: gpd.GeoDataFrame,
    output_dir: str,
    city_name: str,
    gis_loc_col: str,
) -> str:
    """Emit {City}_Change_Detection.geojson: one feature per detected new-
    construction polygon, in WGS84 (EPSG:4326) with [lon, lat] ordering per
    RFC 7946 -- the same convention {City}_Defaulters.geojson uses -- so the
    dashboard map can render both layers with the same pipeline.
    """
    cols_to_keep = [
        c for c in ("Area_sqm", "Est_Bldgs", "GoogleBldg", "City", gis_loc_col, "dist_to_existing")
        if c in joined_gdf.columns
    ]
    gdf = joined_gdf[cols_to_keep + ["geometry"]].copy()
    if gis_loc_col in gdf.columns:
        gdf = gdf.rename(columns={gis_loc_col: "gis_locality"})
    if "City" not in gdf.columns:
        gdf["City"] = city_name

    try:
        gdf = gdf.set_crs(joined_gdf.crs, allow_override=True).to_crs(epsg=4326)
    except Exception as e:
        logger.error(f"Could not reproject change-detection polygons to WGS84: {e}")

    geojson_path = os.path.join(output_dir, f"{city_name}_Change_Detection.geojson")
    if os.path.exists(geojson_path):
        os.remove(geojson_path)  # to_file(driver=GeoJSON) refuses to overwrite
    gdf.to_file(geojson_path, driver="GeoJSON")
    logger.info(f"Change Detection GeoJSON saved: {geojson_path} ({len(gdf):,} features)")
    return geojson_path


def process_change_detection(
    change_shp_path: str,
    gis_gdf: gpd.GeoDataFrame,
    match_register: pd.DataFrame,
    gis_loc_col: str,
    output_dir: str,
    city_name: str
):
    """
    Overlays Earth Engine Change Detection polygons with GIS survey.
    Calculates estimated defaulters per locality using Bucket Math and Locality Crosswalk.
    """
    logger.info(f"Loading Change Detection shapefile from {change_shp_path}...")
    
    try:
        change_gdf = gpd.read_file(change_shp_path)
    except Exception as e:
        logger.error(f"Failed to load Change Detection shapefile: {e}")
        return
        
    if "Est_Bldgs" not in change_gdf.columns:
        logger.error("Change shapefile missing 'Est_Bldgs'. Did you run the updated GEE script?")
        return

    # Ensure CRS match
    if change_gdf.crs != gis_gdf.crs:
        change_gdf = change_gdf.to_crs(gis_gdf.crs)

    # 1. Assign each Change Polygon to the nearest GIS Locality
    logger.info("Spatial Joining new constructions to nearest GIS locality...")
    gis_localities = gis_gdf[[gis_loc_col, "geometry"]].dropna(subset=[gis_loc_col])
    joined = gpd.sjoin_nearest(change_gdf, gis_localities, how="left", distance_col="dist_to_existing")

    # Dashboard map layer: one feature per new-construction polygon (WGS84).
    _write_change_detection_geojson(joined, output_dir, city_name, gis_loc_col)

    # Aggregate
    locality_new_bldgs = joined.groupby(gis_loc_col)["Est_Bldgs"].sum().reset_index()
    locality_new_bldgs.rename(columns={"Est_Bldgs": "Estimated_New_Buildings"}, inplace=True)
    
    # 2. Build the Locality Crosswalk from confident matches
    logger.info("Building Dynamic Locality Crosswalk from matched records...")
    
    # Handle pandas NaN when loaded from CSV
    is_unmatched = match_register["matched_uid"].isna() | (match_register["matched_uid"].astype(str).str.lower().isin(["", "nan", "none"]))
    matched = match_register[~is_unmatched].copy()
    
    # Map each mSeva localityname to the most frequent GIS locality it matched with
    crosswalk = {}
    if not matched.empty and "localityname" in matched.columns and "gis_locality" in matched.columns:
        valid_matches = matched[matched["gis_locality"].astype(str).str.strip() != ""]
        crosswalk = valid_matches.groupby("localityname")["gis_locality"].agg(
            lambda x: x.mode()[0] if not x.mode().empty else None
        ).to_dict()

    # 3. Bucket Math: Subtract unmatched mSeva taxpayers
    logger.info("Applying Locality Bucket subtraction using translated unmatched mSeva records...")
    unmatched_mseva = match_register[is_unmatched].copy()
    
    # Translate unmatched mSeva localities to GIS localities using the crosswalk
    unmatched_mseva["mapped_gis_locality"] = unmatched_mseva["localityname"].map(crosswalk)
    unmatched_mseva["mapped_gis_locality"] = unmatched_mseva["mapped_gis_locality"].fillna(unmatched_mseva["localityname"])
    
    unmatched_counts = unmatched_mseva["mapped_gis_locality"].value_counts().reset_index()
    unmatched_counts.columns = [gis_loc_col, "Unmatched_Paying_Citizens"]
    
    # 4. Merge and Calculate True Defaulters
    final_report = pd.merge(locality_new_bldgs, unmatched_counts, on=gis_loc_col, how="left")
    final_report["Unmatched_Paying_Citizens"] = final_report["Unmatched_Paying_Citizens"].fillna(0)
    
    final_report["Estimated_True_Defaulters"] = final_report["Estimated_New_Buildings"] - final_report["Unmatched_Paying_Citizens"]
    final_report["Estimated_True_Defaulters"] = final_report["Estimated_True_Defaulters"].clip(lower=0)
    
    # Sort by highest defaulters
    final_report = final_report.sort_values(by="Estimated_True_Defaulters", ascending=False)
    
    # Save the outputs
    report_path = os.path.join(output_dir, f"{city_name}_Change_Detection_Summary.csv")
    final_report.to_csv(report_path, index=False)
    
    logger.info(f"Change Detection Summary saved to: {report_path}")
    
    # CLI Output
    total_new = final_report["Estimated_New_Buildings"].sum()
    total_paid = final_report["Unmatched_Paying_Citizens"].sum()
    total_defaulters = final_report["Estimated_True_Defaulters"].sum()
    
    print("\n" + "="*70)
    print("CHANGE DETECTION & NEW COLONIES SUMMARY — " + city_name)
    print("="*70)
    print(f"Total Estimated New Buildings:       {int(total_new):>6,}")
    print(f"(-) Potential Taxpayers (Unmatched): {int(total_paid):>6,}")
    print("-" * 70)
    print(f"Estimated True Defaulters:           {int(total_defaulters):>6,}")
    print("="*70)
    print("Top 5 Localities with New Constructions:")
    for _, row in final_report.head(5).iterrows():
        print(f"  {str(row[gis_loc_col])[:30]:30s}: {int(row['Estimated_True_Defaulters'])} defaulters ({int(row['Estimated_New_Buildings'])} new bldgs)")
    print("="*70 + "\n")
    
    return final_report
