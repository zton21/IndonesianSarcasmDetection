# The Effect of Synthetic Data Generation Using GPT-4o mini on Indonesian Sarcasm Detection Performance

Detecting sarcasm in Indonesian using LLM-generated synthetic data for dataset augmentation.

This project focuses on sarcasm detection in Indonesian by augmenting original data with synthetic samples generated using GPT-4o mini. Three prompting strategies are explored for synthetic data generation: zero-shot with topic guidance, one-shot, and few-shot. The augmented dataset is used to evaluate two transformer-based models: IndoBERT (from IndoNLU) and XLM-RoBERTa. This research aims to assess how synthetic augmentation improves sarcasm detection performance.

## Best Model for Each Dataset

Here is the best-performing model for sarcasm detection on both Twitter and Reddit platforms. You can access the full model via the HuggingFace link.

| Dataset | Model Name                                                                                     | Prompting Technique | #Params |
| ------- | ---------------------------------------------------------------------------------------------- | ------------------- | ------- |
| Twitter | [XLM-R Large](https://huggingface.co/enoubi/XLM-RoBERTa-Twitter-Indonesian-Sarcastic-Few-Shot) | Few-Shot            | 560M    |
| Reddit  | [XLM-R Large](https://huggingface.co/enoubi/XLM-RoBERTa-Reddit-Indonesian-Sarcastic-Few-Shot)  | Few-Shot            | 560M    |

## Dataset

The datasets are available for sarcasm detection on Twitter and Reddit, including preprocessed original data and synthetic data generated using Zero-Shot Topic, One-Shot, and Few-Shot prompting techniques. All links are accessible via Hugging Face.

| Dataset Type            | Twitter                                                                                                               | Reddit                                                                                               |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Original (Preprocessed) | [Twitter-Indonesian-Sarcastic-Fix-Text](https://huggingface.co/datasets/enoubi/Twitter-Indonesian-Sarcastic-Fix-Text) | [Reddit-Indonesian-Sarcastic-Fix-Text](https://huggingface.co/datasets/enoubi/Reddit-Indonesian-Sarcastic-Fix-Text) |
| Zero-Shot Topic         | [Twitter-Indonesian-Sarcastic-Synthetic-Zero-Shot-Topic](https://huggingface.co/datasets/enoubi/Twitter-Indonesian-Sarcastic-Synthetic-Zero-Shot-Topic) | [Reddit-Indonesian-Sarcastic-Synthetic-Zero-Shot-Topic](https://huggingface.co/datasets/enoubi/Reddit-Indonesian-Sarcastic-Synthetic-Zero-Shot-Topic) |
| One-Shot                | [Twitter-Indonesian-Sarcastic-Synthetic-One-Shot](https://huggingface.co/datasets/enoubi/Twitter-Indonesian-Sarcastic-Synthetic-One-Shot) | [Reddit-Indonesian-Sarcastic-Synthetic-One-Shot](https://huggingface.co/datasets/enoubi/Reddit-Indonesian-Sarcastic-Synthetic-One-Shot) |
| Few-Shot                | [Twitter-Indonesian-Sarcastic-Synthetic-Few-Shot](https://huggingface.co/datasets/enoubi/Twitter-Indonesian-Sarcastic-Synthetic-Few-Shot) | [Reddit-Indonesian-Sarcastic-Synthetic-Few-Shot](https://huggingface.co/datasets/enoubi/Reddit-Indonesian-Sarcastic-Synthetic-Few-Shot) |
