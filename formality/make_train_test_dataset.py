"""Make dataset for formality classification."""

import pickle
import click
import random
import json
@click.command()
@click.option("-i", "--raw-dataset-path", default="./data/formality-corpus/", show_default=True, help="Path to Raw Dataset")
@click.option("-o", "--output-path", default="./data/formality-corpus/", show_default=True, help="Output Data Path")
def main(raw_dataset_path, output_path):
    """Make train, dev and test dataset."""
    scores = []
    sentences = []
    
    answers = raw_dataset_path + '/answers'
    blog = raw_dataset_path + '/blog'
    news = raw_dataset_path + '/news'
    email = raw_dataset_path + '/email'
    categories = [answers, blog, news, email]
    
    print("Reading raw dataset files")
    for cat in categories:
        print("{:20s}  {:20s}".format("Reading File: ", cat))
        with open(cat, 'r', encoding='utf-8', errors= 'ignore') as f:
            for line in f:
                elements = line.strip().split('\t')
                scores.append(float(elements[0].strip()))
                sentences.append(elements[3].strip())
    print("Finished Reading Files")

    data = list(zip(scores, sentences))
    random.shuffle(data)

    scores, sentences = zip(*data)
    scores = [0 if s <0 else 1 for s in scores]
    test_data = []
    for i in range(1000):
        test_data.append({'label':scores[i], 'sentence':sentences[i]})
    """
    test_data['scores'] = scores[:1000]
    test_data['sentences'] = sentences[:1000]
    """

    dev_data = []
    for i in range(1000, 4000):
        dev_data.append({'label':scores[i], 'sentence':sentences[i]})
    """
    dev_data['scores'] = scores[1000:4000]
    dev_data['sentences'] = sentences[1000:4000]
    """
    train_data = []
    for i in range(4000, len(scores)):
        train_data.append({'label':scores[i], 'sentence':sentences[i]})
    """
    train_data['scores'] = scores[4000:]
    train_data['sentences'] = sentences[4000:]
    """
    print("Length of train data: ", len(train_data))

    print("Dumping train, dev, test files")
    with open(output_path + '/train.json', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item)+'\n')
        #pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

    with open(output_path + '/dev.json', 'w') as f:
        for item in dev_data:
            f.write(json.dumps(item)+'\n')
        #pickle.dump(dev_data, f, pickle.HIGHEST_PROTOCOL)

    with open(output_path + '/test.json', 'w') as f:
        for item in test_data:
            f.write(json.dumps(item)+'\n')
        #pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
