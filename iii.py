import custom_tag_config
from custom_tag_config import custom_tags

tag_set = ['817725626636951440', '8353647472807540595', '8531798581027376660', '8624511174253220625', '8687593093433542582', '8688506092654802735', '8931775978304697812', '9096057992187323206', '9293129351730695055', '9364161469948028785', '988882105507299009']
print("l1 = ",len(tag_set))
print(custom_tags)
custom_tags[0]['vocab_fun'](tag_set)
embedding_matrix = custom_tags[0]['initializer_function']
print(custom_tag_config.embedding_size) #768
print(custom_tag_config.vocab_size) #12
print(custom_tag_config.vocab)