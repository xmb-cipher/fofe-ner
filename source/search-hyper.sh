#!/cs/local/bin/bash


# rm -rf log/learning-rate log/sample-rate log/feature-choice


# tensorflow ner-trainer-conll2003.py word2vec/reuters256 processed-data \
# 		--feature_choice=63 --n_batch_size=512 --learning_rate=0.512 --word_alpha=0.5 --max_iter=16 \
# 		--overlap_rate=0.64 --disjoint_rate=0.16 --dropout=True | tee -a learning-rate



# for lr in 0.4096 0.2048 0.1024 0.0512 0.0256 0.0128 0.0064
# do
# 	rm -rf *.predicted *.error
# 	printf "\nlearning-rate: %f \n" $lr | tee -a log/learning-rate
# 	tensorflow ner-trainer-conll2003.py word2vec/reuters256 processed-data \
# 		--feature_choice=63 --n_batch_size=512 --learning_rate=$lr --word_alpha=0.5 --max_iter=16 \
# 		--overlap_rate=0.64 --disjoint_rate=0.16
# 	python CoNLL2003eval.py processed-data/eng.testa valid.predicted | conlleval | tee -a log/learning-rate
# 	python CoNLL2003eval.py processed-data/eng.testb test.predicted  | conlleval | tee -a log/learning-rate
# 	python CoNLL2003eval.py processed-data/eng.train train.predicted | conlleval | tee -a log/learning-rate
# 	printf "\n\n" | tee -a log/learning-rate
# done


for (( i = 32; i <= 32; i++ )) do
	rm -rf *.predicted *.error
	printf "\nchoice: %i \n" $i | tee -a log/feature-choice
	tensorflow ner-trainer-conll2003.py word2vec/reuters256 processed-data \
		--feature_choice=$i --n_batch_size=512 --learning_rate=0.1024 --word_alpha=0.5 --max_iter=16 \
		--overlap_rate=0.64 --disjoint_rate=0.16
	python CoNLL2003eval.py processed-data/eng.testa valid.predicted | conlleval | tee -a log/feature-choice
	python CoNLL2003eval.py processed-data/eng.testb test.predicted  | conlleval | tee -a log/feature-choice
	python CoNLL2003eval.py processed-data/eng.train train.predicted | conlleval | tee -a log/feature-choice
	printf "\n\n" | tee -a log/feature-choice
done


# for overlap in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# do
# 	for disjoint in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# 	do
# 		rm -rf *.predicted *.error
# 		printf "\noverlap-rate: %f, disjoint-rate: %f \n" $overlap $disjoint | tee -a log/sample-rate
# 		tensorflow ner-trainer-conll2003.py word2vec/reuters256 processed-data \
# 			--feature_choice=63 --n_batch_size=512 --learning_rate=0.1024 --word_alpha=0.5 --max_iter=16 \
# 			--overlap_rate=$overlap --disjoint_rate=$disjoint
# 		python CoNLL2003eval.py processed-data/eng.testa valid.predicted | conlleval | tee -a log/sample-rate
# 		python CoNLL2003eval.py processed-data/eng.testb test.predicted  | conlleval | tee -a log/sample-rate
# 		python CoNLL2003eval.py processed-data/eng.train train.predicted | conlleval | tee -a log/sample-rate
# 		printf "\n\n" | tee -a log/sample-rate
# 	done
# done



