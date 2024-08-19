# semantic-router-classifier

Example of text-classifier based on [semantic-router](https://github.com/aurelio-labs/semantic-router) library and OpenAI embedding model

## Usage

1. Installation
   > Requires python v3.10 or greater
   To install dependencies (linux) run:

   ```sh
   python3 -m venv .venv && source .venv/bin/activate
   pip install -U pip setuptools
   pip install .
   ```

2. Load dataset
    To create dataset you can use [example file](./classifier/semantic_router/files/text_topics_example.xlsx). After creating your dataset, call file `text_topics_optimized.xlsx` and store it near by origin.

3. Run tests for classifier
    `python test_classifier.py`

4. Use in your code
    > This is not pypi module yet
    For using classifier in your code you need to create classifier instance and call `classify_text` function

    ```python
    from classifier.semantic_router.topic_classifier import Classifier

    question = "your question"
    classifier = Classifier()
    classifier.classify_text(question)
    ```

## Code quality

To run linters run `pip install .[code-quality]` then laungh `. ./code_quality_check.sh`
