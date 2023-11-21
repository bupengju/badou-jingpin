def predict(text):
    # Load the model
    model = load_model()
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")
    # Generate the output
    outputs = model.generate(**inputs)
    # Decode the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output