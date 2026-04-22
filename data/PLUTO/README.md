# census_tract_attributes_from_pluto2015.csv

This document describes the inferred feature definitions for `census_tract_attributes_from_pluto2015.csv`. The preprocessing pipeline is not explicitly documented by the original authors, so some interpretations here may not perfectly reflect their exact methodology. For authoritative field definitions, refer to `pluto_datadictionary.pdf`. For the underlying raw data, see `nyc_pluto_15v1.zip`.

Census-tract-level features derived from NYC's [PLUTO (Primary Land Use Tax Lot Output)](https://www1.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page) dataset, version 15v1 (June 2015). Each row represents one census tract.

---

## Preprocessing

All columns except `far_builtfar_avg` and `far_builtfar_std` are **Location Quotient (LQ)** transformed. Raw lot-level counts are first aggregated per census tract, then normalized as:

```
LQ(feature, tract) = (feature_share_in_tract) / (feature_share_citywide)
```

| LQ value     | Interpretation                                                 |
| ------------ | -------------------------------------------------------------- |
| `0.0`        | Building class / land use is completely absent from this tract |
| `0 < LQ < 1` | Present but under-represented relative to the NYC average      |
| `= 1.0`      | Exactly at the NYC average concentration                       |
| `> 1.0`      | Over-represented — more concentrated than the NYC average      |

High LQ values (e.g., 50–400) are expected for rare building types (hospitals, transportation facilities) that cluster heavily in specific tracts.

---

## Column Reference

### Identifier

| Column       | Description                                                                  |
| ------------ | ---------------------------------------------------------------------------- |
| `BoroCT2010` | Borough + Census Tract 2010 identifier (e.g., `1000100` = Manhattan tract 1) |

---

### Building Class (`bldgclass_*`)

LQ of buildings belonging to each NYC building class letter. The raw PLUTO field `BldgClass` is a 2-character code (e.g., `A1`, `D3`); here only the first letter (major class) is used.

| Column        | Building Class                                          |
| ------------- | ------------------------------------------------------- |
| `bldgclass_A` | One-family dwellings                                    |
| `bldgclass_B` | Two-family dwellings                                    |
| `bldgclass_C` | Walk-up apartments                                      |
| `bldgclass_D` | Elevator apartments                                     |
| `bldgclass_E` | Warehouses                                              |
| `bldgclass_F` | Factory and industrial buildings                        |
| `bldgclass_G` | Garages and gasoline stations                           |
| `bldgclass_H` | Hotels                                                  |
| `bldgclass_I` | Hospitals and health facilities                         |
| `bldgclass_J` | Theatres                                                |
| `bldgclass_K` | Store buildings (taxpayers)                             |
| `bldgclass_L` | Loft buildings                                          |
| `bldgclass_M` | Religious facilities (churches, synagogues, etc.)       |
| `bldgclass_N` | Asylums and homes                                       |
| `bldgclass_O` | Office buildings                                        |
| `bldgclass_P` | Places of public assembly and cultural facilities       |
| `bldgclass_Q` | Outdoor recreation facilities                           |
| `bldgclass_R` | Condominiums                                            |
| `bldgclass_S` | Residence — multiple use                                |
| `bldgclass_T` | Transportation facilities (airports, piers, etc.)       |
| `bldgclass_U` | Utility bureau properties (bridges, tunnels, utilities) |
| `bldgclass_V` | Vacant land                                             |
| `bldgclass_W` | Educational structures                                  |
| `bldgclass_Y` | Selected government installations                       |
| `bldgclass_Z` | Miscellaneous                                           |

---

### Land Use Category (`landuse_*`)

LQ of tax lots in each of NYC's 11 DCP land use categories. Derived from the PLUTO `LandUse` field.

| Column       | Land Use Category                        |
| ------------ | ---------------------------------------- |
| `landuse_1`  | One & two family buildings               |
| `landuse_2`  | Multi-family walk-up buildings           |
| `landuse_3`  | Multi-family elevator buildings          |
| `landuse_4`  | Mixed residential & commercial buildings |
| `landuse_5`  | Commercial & office buildings            |
| `landuse_6`  | Industrial & manufacturing               |
| `landuse_7`  | Transportation & utility                 |
| `landuse_8`  | Public facilities & institutions         |
| `landuse_9`  | Open space & outdoor recreation          |
| `landuse_10` | Parking facilities                       |
| `landuse_11` | Vacant land                              |

---

### Area Ratios (`landarearatio_*`)

Ratio of each floor area type to total lot area, aggregated across the tract. Derived from PLUTO area fields divided by `LotArea`.

| Column                     | Description                             |
| -------------------------- | --------------------------------------- |
| `landarearatio_bldgarea`   | Total building floor area / lot area    |
| `landarearatio_comarea`    | Commercial floor area / lot area        |
| `landarearatio_factryarea` | Factory/warehouse floor area / lot area |
| `landarearatio_garagearea` | Garage floor area / lot area            |
| `landarearatio_lotarea`    | Lot area (normalization reference)      |
| `landarearatio_officearea` | Office floor area / lot area            |
| `landarearatio_otherarea`  | Other floor area / lot area             |
| `landarearatio_resarea`    | Residential floor area / lot area       |
| `landarearatio_retailarea` | Retail floor area / lot area            |
| `landarearatio_strgearea`  | Storage/loft floor area / lot area      |

---

### Object Density (`numobj_*`)

Counts of physical objects normalized by census tract area (per m²).

| Column                  | Description                                        |
| ----------------------- | -------------------------------------------------- |
| `numobj_numbldgs_/m2`   | Number of buildings per m²                         |
| `numobj_numfloors_/m2`  | Number of floors per m²                            |
| `numobj_unitsres_/m2`   | Residential units per m²                           |
| `numobj_unitstotal_/m2` | Total units (residential + non-residential) per m² |

---

### Construction Era (`yearbuilt_*`)

LQ of buildings completed in each decade. Derived from the PLUTO `YearBuilt` field.

| Column            | Era             |
| ----------------- | --------------- |
| `yearbuilt_1900s` | Built 1900–1919 |
| `yearbuilt_1920s` | Built 1920–1929 |
| `yearbuilt_1930s` | Built 1930–1939 |
| `yearbuilt_1940s` | Built 1940–1949 |
| `yearbuilt_1950s` | Built 1950–1959 |
| `yearbuilt_1960s` | Built 1960–1969 |
| `yearbuilt_1970s` | Built 1970–1979 |
| `yearbuilt_1980s` | Built 1980–1989 |
| `yearbuilt_1990s` | Built 1990–1999 |
| `yearbuilt_2000s` | Built 2000–2009 |
| `yearbuilt_2010s` | Built 2010–2015 |

---

### Designation Flags (`specificarea_*`)

Proportion of lots in the tract that carry a historic or landmark designation, as recorded by the NYC Landmarks Preservation Commission.

| Column                  | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| `specificarea_histdist` | Proportion of lots within a designated NYC historic district |
| `specificarea_landmark` | Proportion of lots designated as individual NYC landmarks    |

---

### Floor Area Ratio (`far_*`)

Built FAR = total building floor area / lot area. These columns are **not** LQ-transformed; they are raw averages across lots in the tract.

| Column             | Description                                              |
| ------------------ | -------------------------------------------------------- |
| `far_builtfar_avg` | Mean built FAR across lots in the tract                  |
| `far_builtfar_std` | Standard deviation of built FAR across lots in the tract |

---

## Cautions

**LQ values are unbounded above.** There is no ceiling on LQ. A value of 300 is valid and simply means that building type is 300× more concentrated in that tract than citywide. Do not treat these as probabilities or percentages.

**`0.0` is not the same as "average."** A zero means the feature is completely absent from the tract. LQ = 1.0 is the average. Features that are absent in a tract but present elsewhere will always produce exactly `0.0`, not a small positive value.

**Building class and land use are correlated.** The `bldgclass_*` and `landuse_*` features overlap significantly in what they encode (PLUTO derives land use directly from building class). Including both in a model without care may introduce redundancy.

**Year built may be estimated.** PLUTO flags some `YearBuilt` values as estimates (`BuiltCode = E`). The `yearbuilt_*` columns inherit this uncertainty.

**Floor area fields have known gaps.** PLUTO notes that office, retail, garage, storage, and factory areas are not available for one-, two-, or three-family structures. The `landarearatio_*` columns derived from these fields may undercount in predominantly residential tracts.

**Snapshot in time.** This file reflects the state of NYC land use as of June 2015. It does not capture construction, demolition, or rezoning after that date.

**FAR is an estimate.** Per the PLUTO data dictionary, `BuiltFAR` is described as an _estimate_ based on rough building area and lot area measurements. The `far_builtfar_avg` and `far_builtfar_std` columns inherit this imprecision.
