from flair.datasets import UD_ENGLISH
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

corpus = UD_ENGLISH()
print(corpus)

label_type = 'upos'

label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

embedding_types = [
    TransformerWordEmbeddings('xlnet-base-cased',
                              layers="-1",
                              subtoken_pooling="first",
                              fine_tune=True,
                              use_context=False)    
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=False,
                        use_rnn=False,
                        reproject_embeddings=False)

trainer = ModelTrainer(tagger, corpus)

trainer.fine_tune('resources/taggers/xlnet-upos-adam',
              learning_rate=5.0e-6,
              mini_batch_size=16,
              )

