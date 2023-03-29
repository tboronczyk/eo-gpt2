
## Steps

1. **Download corpus texts**

        # download Tekstaro texts
        python download-tekstaro.py \
          --tmp_dir=./tmp \
          --output_dir=./corpus

        # download Wikipedia texts
        python download-wikipedia.py \
          --page_list=./wikipedia-featured.txt \
          --output_dir=./corpus
        
        # download Marvirinstrato
        wget -O ./corpus/marvirinstrato.txt \
            https://www.smashwords.com/books/download/267558/6/latest/0/0/marvirinstrato-originalaj-noveloj-en-esperanto-esperanto-edi.txt

        # download OSCAR text
        wget -O ./corpus/oscar.eo.txt \
            https://cdn-datasets.huggingface.co/EsperBERTo/data/oscar.eo.txt

  The corpus consists of text from the following sources:

 * [Tekstaro](https://tekstaro.com/elshuti.html) excluding _Homaranismo (1906)_
 * Wikipedia [Elstaraj artikoloj](https://eo.wikipedia.org/wiki/Kategorio:Elstaraj_artikoloj)
 * [Marvirinstrato](https://www.smashwords.com/books/view/267558)
 * Esperanto subset of [OSCAR](https://oscar-project.org/)

2. **Compile the corpus**

        python compile-corpus.py --split --split_len=2048

3. **Train the tokenizer**

        python train-tokenizer.py --init --train_file=corpus.txt

3. **Split training and test data**

        python split.py

4. **Train the model**

        python train.py --init --epocs=1

5. **Prompt the model**

        python inference.py --text="Saluton"

