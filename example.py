"""Test with word embeddings."""
from reach import Reach
from plate.plate import circular_convolution, decode


if __name__ == "__main__":

    r = Reach.load("PATH_TO_EMBEDDINGS")

    # Encode "dog chase cat"
    a = circular_convolution(r["subject"], r["dog"])
    b = circular_convolution(r["verb"], r["chase"])
    c = circular_convolution(r["object"], r["cat"])

    sentence = a + b + c
    vec = decode(r["subject"], sentence)
    result = r.nearest_neighbor(vec)

    # The top result should be dog
