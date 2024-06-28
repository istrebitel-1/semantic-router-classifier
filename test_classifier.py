import asyncio

from classifier.tests.test_classifier import test_topic_classifier


if __name__ == "__main__":
    asyncio.run(test_topic_classifier(skip_assertion=True))
