# LayaLink

**LayaLink** is a fine-tuned decision model for concept mapping, ontology alignment, and entity linking. Built on [Laya](https://huggingface.co/convaiinnovations/laya), it evaluates candidate concepts and produces structured, task-specific decisions with associated probabilities.

This repository contains the source code, training scripts, and supporting resources used to develop LayaLink.

**Model:** [sensein/layalink](https://huggingface.co/sensein/layalink)   

## Supported Tasks

LayaLink supports three tasks:

| Task | Description | Decision outputs |
|---|---|---|
| **Ontology Alignment** (`ontology_alignment`) | Determines the semantic relationship between a source and target ontology concept using their labels, definitions, parent concepts, and ontology context. | `relation`, `exact_match` |
| **Entity Linking** (`entity_linking`) | Evaluates whether a candidate ontology concept matches an entity mention in its textual context. | `mapping`, `accept` |
| **Concept Selection** (`concept_selection`) | Selects the most appropriate concept from a supplied list of candidates for a given entity mention. | `concept`, `resolvable` |


## License

LayaLink is released under the **Apache License 2.0**.
