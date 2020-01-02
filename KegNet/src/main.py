"""
Knowledge Extraction with No Observable Data (NeurIPS 2019)
"""
from kegnet.generator.train import main as train_generator


def main():
    n_generators = 5
    dataset = 'mnist'
    path_teacher = f'../pretrained/{dataset}.pth.tar'
    path_out = f'../out/{dataset}'

    generators = []
    for i in range(n_generators):
        path_gen = f'{path_out}/generator-{i}'
        path_model = train_generator(dataset, path_teacher, path_gen, i)
        generators.append(path_model)


if __name__ == '__main__':
    main()

