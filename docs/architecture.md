# Planned architecture

This is a scaffold, with no implemented data pipeline.

Raw CCTV images ? engineering ? interim observations ? validated processed data
? analysis ? dashboard.

Approved external sources ? scraping ? raw external data ? preparation and
integration with traffic observations. Integration rules remain to be agreed.

Engineering owns image ingestion, detection, preprocessing, and validation.
Scraping owns source definitions and collection. Analysis owns exploratory,
temporal, and spatial analysis. The dashboard will read processed outputs.
Configuration lives in configs/config.yaml; paths resolve from the repository root.

Use Parquet for planned structured outputs. Agree on schema, time semantics,
camera metadata, and join keys before implementation. The provisional dictionary
is a proposal, not a claim about available source data.

Scripts provide future entry points. Functions deliberately raise
NotImplementedError. The Streamlit landing page is the only implemented UI.
