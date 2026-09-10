"""
matchers — Sub-package containing the matching stages.

Each matcher module implements one step of the layered matching
pipeline (spatial, mobile, property-ID, name+locality).  Import
individual matchers from this package as needed::

    from pipeline.matchers.spatial import SpatialMatcher
    from pipeline.matchers.mobile import MobileMatcher
    from pipeline.matchers.property_id import PropertyIdMatcher
    from pipeline.matchers.name_locality import NameLocalityMatcher
"""
