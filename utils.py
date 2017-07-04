import numpy as np

def load_data(x_path, y_path):
    x = []
    y = []
    for line in open(x_path, 'rb'):
        x_d = []
        sens = line.strip().split('<sssss>')
        for sen in sens:
            if sen.strip() != '':
                x_s = []
                words = sen.strip().split(' ')
                for word in words:
                    x_s.append(int(word))
                x_d.append(x_s)
        x.append(x_d)
    for line in open(y_path, 'rb'):
        label = line.strip()
        y.append(int(label) - 1)
    return np.array(x), np.array(y)

def load_embedding(path):
    embedding_matrix = []
    for line in open(path, 'rb'):
        embedding = []
        vector = line.strip().split(' ')
        for val in vector:
            embedding.append(float(val))
        embedding_matrix.append(embedding)
    return np.array(embedding_matrix)

def load_data2(x_path, y_path):
    x = []
    y = []
    for line in open(x_path, 'rb'):
        x_d = []
        words = line.strip().split(' ')
        for word in words:
            x_d.append(int(word))
        x.append(x_d)
    for line in open(y_path, 'rb'):
        label = line.strip()
        y.append(int(label))
    return np.array(x), np.array(y)

def load_x(x_path):
    x = []
    for line in open(x_path, 'rb'):
        x_d = []
        words = line.strip().split(' ')
        for word in words:
            x_d.append(int(word))
        x.append(x_d)
    return np.array(x)

def load_embedding2(path):
    embedding_matrix = []
    for line in open(path, 'rb'):
        embedding = []
        vector = line.strip().split(' ')
        for val in vector:
            embedding.append(float(val))
        embedding_matrix.append(embedding)
    return np.array(embedding_matrix)

def batch(inputs):
    batch_size = len(inputs)

    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()

    sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
    sentence_size = max(map(max, sentence_sizes_))

    b = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32)  # == PAD

    sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    for i, document in enumerate(inputs):
        for j, sentence in enumerate(document):
            sentence_sizes[i, j] = sentence_sizes_[i][j]
            for k, word in enumerate(sentence):
                b[i, j, k] = word

    return b, document_sizes, sentence_sizes

def batch2(inputs):
    batch_size = len(inputs)

    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()

    # sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
    # sentence_size = max(map(max, sentence_sizes_))

    b = np.zeros(shape=[batch_size, document_size], dtype=np.int32)  # == PAD

    #sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    for i, document in enumerate(inputs):
        for j, word in enumerate(document):
            # sentence_sizes[i, j] = sentence_sizes_[i][j]
            # for k, word in enumerate(sentence):
            b[i, j] = word

    return b, document_sizes