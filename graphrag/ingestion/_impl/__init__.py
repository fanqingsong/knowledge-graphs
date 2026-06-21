"""graphrag.ingestion._impl — private implementation of the ingestion pipeline.

All stage implementations (`Cleaner` / `Chunker` / `ChunkEmbedder` /
`GraphMiner`), the `Stage` protocol, the `Ingestor` base + `LocalIngestor`,
and the internal conversion helpers (`_converters`) live here.

External code must import the public names from `graphrag.ingestion`, never
from this subpackage. The leading underscore is the contract.
"""
