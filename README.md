# Multimodal_sentiment_analysis

## Phase 1 completed

**Professional Project Setup**: structured the project with a clean, scalable repository layout, ready for more complex features.

**End-to-End Data Pipeline**: Built a complete, version-controlled data pipeline which can programmatically fetch data from the Twitter API, preprocess it, and track both the raw and cleaned datasets using **DVC**, with the data safely backed up to Google Drive.

**Baseline Model Training**: Successfully fine-tuned a pre-trained Transformer model (DistilBERT) on a standard benchmark dataset (IMDB).

**API Deployment**: Served the trained model via a live API using **FastAPI**, allowing it to be accessed by other applications. Also tested this live endpoint.

**MLOps and Troubleshooting**: Most importantly, navigated and solved real-world MLOps challenges, including:

* Managing large files that don't belong in Git.
* Correcting Git history after accidental commits.
* Handling API credentials and secrets securely.

We now have a fully functional, end-to-end "scaffolding" for a machine learning application. The next phases will focus on significantly improving each component of this foundation.

## Phase 2: Advanced Data & Models. ( CURRENT )

The goal of this phase is to move beyond a single data source and a simple model. We will introduce a more diverse dataset to improve the model's real-world performance and integrate a more powerful Transformer architecture. We will also begin laying the groundwork for the Ethical AI framework by introducing a bias detection toolkit.
