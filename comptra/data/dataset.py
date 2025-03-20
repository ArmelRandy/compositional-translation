from datasets import load_dataset
from comptra.languages import *

def get_datasets(
    dataset_name_or_path: str,
    language: str
):
    """
    Get a dataset given its description and the language of interest
    Arguments
    ---------
        - dataset_name_or_path: str,
            Description of the dataset of interest
        - language: str,
            Language of interest (e.g. English)
    Examples
    --------
    >>> get_datasets("flores", "English")
    DatasetDict({
        dev: Dataset({
            features: ['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink', 'sentence'],
            num_rows: 997
        })
        devtest: Dataset({
            features: ['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink', 'sentence'],
            num_rows: 1012
        })
    })
    """
    if dataset_name_or_path == "flores":
        if language in NON_FLORES:
            from comptra.data.extension import get_datasets as get_extension_datasets
            dataset = get_extension_datasets(MAPPING_LANG_TO_KEY[language])
        else:
            dataset = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])
    elif dataset_name_or_path == "ntrex":
        from comptra.data.ntrex import get_datasets as ntrex
        code = MAPPING_LANG_TO_KEY_NTREX[language]
        dataset = ntrex(code, code)[0]
    elif dataset_name_or_path == "tico":
        from comptra.data.tico import get_datasets as tico
        if language == "English":
            #dataset, _ = tico("English", "Hausa")
            dataset, _ = tico("English", "Bengali")
        else:
            _, dataset = tico("English", language)
    elif dataset_name_or_path == "ood":
        # dev = Flores, devtest = TICO
        from comptra.data.tico import get_datasets as tico
        # FLORES-200
        if language in NON_FLORES:
            from comptra.data.extension import get_datasets as get_extension_datasets
            flores_dataset = get_extension_datasets(MAPPING_LANG_TO_KEY[language])
        else:
            flores_dataset = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])
        # TICO-19
        if language == "English":
            #dataset, _ = tico("English", "Hausa")
            dataset, _ = tico("English", "Bengali")
        else:
            _, dataset = tico("English", language)
        # dev = Flores, devtest = TICO
        dataset["dev"] = flores_dataset["dev"]
    else:
        raise ValueError(f"Unsupported dataset description '{dataset_name_or_path}")
    return dataset

