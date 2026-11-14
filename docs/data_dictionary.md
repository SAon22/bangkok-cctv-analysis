# Provisional traffic dataset schema

**Proposal only.** No dataset has been collected or produced. Field types,
nullability, categories, and time conventions must be confirmed against real
sources before implementation.

Proposed grain: one observation per camera per captured image, with
(camera_id, timestamp) as a candidate key. Confirm whether simultaneous images
require an additional image identifier.

| Field | Proposed type | Proposed meaning |
| --- | --- | --- |
| camera_id | string | Stable identifier from agreed camera metadata |
| timestamp | timezone-aware datetime | Image capture time; store in UTC and display in Asia/Bangkok |
| car_count | nonnegative integer | Cars detected in the image |
| motorcycle_count | nonnegative integer | Motorcycles detected in the image |
| bus_count | nonnegative integer | Buses detected in the image |
| truck_count | nonnegative integer | Trucks detected in the image |
| total_vehicle_count | nonnegative integer | Sum of the four agreed vehicle categories |

Proposed validation: required fields are non-null; counts are integral and
nonnegative; timestamps have explicit time zones; candidate keys are unique.
Missing or failed detections must not silently become zero counts.

Counts represent vehicles visible in a single image, not unique vehicles or
traffic flow across images. Category definitions and model label mappings remain
to be agreed. Camera coordinates and external-data join fields may be added only
after source availability and provenance are verified.
