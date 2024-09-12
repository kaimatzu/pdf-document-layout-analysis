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

    # Get the logits and apply a sparsity-inducing function (ReLU)
    logits = outputs.logits[0]
    sparse_values = torch.nn.functional.relu(logits).cpu().numpy()

    # Create sparse output with indices and values
    indices = sparse_values.nonzero()  # This returns a tuple of arrays
    values = sparse_values[indices]

    # Convert indices tuple to a list of lists and values to Python native types
    indices_list = [list(ind) for ind in zip(*indices)]  # Each index pair is retained [token_position, vocab_index]
    
    # Convert numpy values to Python floats
    values_list = [float(v) for v in values]

    # Prepare the sparse representation as a dictionary
    sparse_representation = {
        "indices": indices_list,  # List of [token_position, vocab_index]
        "values": values_list      # Corresponding values
    }

    return sparse_representation
