from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load the SPLADE model and tokenizer directly from Hugging Face
def load_splade_model():
    tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
    model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
    return tokenizer, model

# Generate SPLADE embeddings for a given text
def generate_splade_embeddings(text):
    tokenizer, model = load_splade_model()

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Forward pass through the SPLADE model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits and apply a sparsity-inducing function (ReLU, etc.)
    logits = outputs.logits[0]
    sparse_values = torch.nn.functional.relu(logits).cpu().numpy()

    # Create sparse output with indices and values
    indices = sparse_values.nonzero()  # This returns a tuple of arrays
    values = sparse_values[indices]

    # Convert indices tuple to a list of lists, ensuring all values are native Python int
    indices_list = list(zip(*indices))
    indices_list = [list(map(int, ind)) for ind in indices_list]  # Convert all indices to native Python ints

    # Convert values to native Python floats
    values_list = values.astype(float).tolist()

    # Prepare the sparse representation as a dictionary
    sparse_representation = {
        "indices": indices_list,  # List of indices
        "values": values_list      # Corresponding values
    }

    return sparse_representation
