def lex_order_new(reviews, input):
    words = set()
    # print(f"Unique words in lex_order_new vocabulary: {len(words)}")
    reviews = reviews + input
    for i in reviews: # problem: crashed because I traid to make another list of lists of it so I can maintain the order but that was too much for memory 1/3
        for n in i.split(): # solution found: just use the input reviews as the order and don't make a list of lists 1/7
            words.add(n)
    words = sorted(list(words))
    hashing = {}
    order = 1
    for i in words:
        hashing[i] = order
        order += 1

    return hashing