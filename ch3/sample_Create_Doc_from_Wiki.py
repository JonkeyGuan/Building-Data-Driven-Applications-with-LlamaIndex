from llama_index.readers.wikipedia import WikipediaReader

loader = WikipediaReader() 
documents = loader.load_data(
    pages=['Pythagorean theorem','General relativeity']
) 
print(f"loaded {len(documents)} documents")
