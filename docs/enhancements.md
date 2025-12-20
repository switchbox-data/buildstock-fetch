# Buildstock Fetch Enhancements

A reference for additional data enhancements from Switchbox.

## Table of Contents
- [Background](#background)


## Additional upgrade scenarios

In addition to the upgrade scenarios directly available from [NREL's S3 database](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2F), `buildstock-fetch` offers a handful of additional upgrade scenarios for the 2024 ResStock release. These include:

1. Upgrade 17: ENERGY STAR heat pump with elec backup + Full Appliance Electrification with Efficiency
2. Upgrade 18: High efficiency cold-climate air-to-air heat pump with electric backup + Full Appliance Electrification with Efficiency
3. Upgrade 19: Ultra high efficiency heat pump with elec backup + Full Appliance Electrification with Efficiency
4. Upgrade 20: ENERGY STAR heat pump with existing system as backup + Full Appliance Electrification with Efficiency
5. Upgrade 21: Geothermal heat pump + Full Appliance Electrification with Efficiency
6. Upgrade 22: Full Appliance Electrification with Efficiency Only

These upgrades were calculated by combining parts of different upgrades. The table below shows the composition of each upgrade scenario.

| Upgrade ID | HVAC  | Appliances | Envelope |
|------------|--------------|--------------|------------|
| 17         | 1            | 1            | 1         |
| 18         | 2            | 2            | 2         |
| 19         | 3            | 3            | 3         |
| 20         | 4            | 4            | 4         |
| 21         | 5            | 5            | 5         |
| 22         | 0            | 0            | 0         |

More detail on which specific columns correspond to the functional groups (e.g. HVAC, appliances, envelope) above, see file: `buildstock_fetch/data/load_curve_column_map/data_dictionary_2024_load_curve_labeled.csv`.
