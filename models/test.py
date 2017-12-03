def test(a=2, b=3):
    print(a, b)


inputs = {'a': 1, 'b': 1}
test(**inputs)
