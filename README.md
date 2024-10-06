<h1>Dataset Source</h1>
    <p>This dataset is sourced from Data Engineering team.</p>

<h1>Data Cleaning</h1>
    <ol>
        <li>Firstly presence of null value is checked in the dataset.<br>
            <p>
            <img src= "documentation\data_cleaning\null_check.png" alt = "Null value checking" width = "500"><br>
            As the model is designed to performing scementic search with the help of product name only so in case of null value in the product name they are dropped, and the null value in other features are not handled as they are used during display of product detail.<br>
            </p>
        </li>
        <li>Drop duplicate names in the dataset.<br>
            <p>
            Duplicate product names are dropped.
            </p>
        </li>
        <li>Normalization<br>
            <p>
            The text data of product name is all converted to lower case.
            </p>
        </li>
        <li>Removing special characters, punctuations and redundent white space.<br>
            <p>
            Regular expression matching operation was used to remove special characters, punctuations and redundent white space along with presence of single characters that were left after removals of special characters.
            </p>
        </li>
        <li>Removing Stop Words.<br>
            <p>
            Spacy liberary was used to remove the commenly known stopwords like a, an, the, but, if, etc that are predefined in spacy liberary.
            </p>
        </li>  
    </ol>

<h1>Preprocessing</h1>
    <ol>
        <li>Tokenization and vector embedding.<br>
            <p>
            Pretrained bert tokenizer and bert model were used to perform word level tokenization and vector embedding. Which are added to orginal dataframed as numpy and the dataframe is saved in pkl format to preserve the numpy array.
            </p>
        </li>
    </ol>

<h1>Exploratory Data Analysis</h1>
    <ol>
        <li>Basic Descriptive Statics<br>
            <p>
            When checked for frequency of uniquie product names after data cleaning.<br>
            <img src = "documentation\eda\statics.png" alt = "Product Frequency" width = "500">
            </p>
        </li>
        <li>Word Frequency Statics<br>
            <p>
            When checked for unique words present in the dataset it is observed to have 557 unique words.<br>
            <img src = "documentation\eda\word_frequency_557.png" alt = "Word Frequency" width = "500">
            </p>
        </li>
        <li>Word Cloud<br>
            <p>
            Plot of wordcloud is shown below.<br>
            <img src = "documentation\eda\wordcloud.png" alt = "Word Cloud" width = "500">
            </p>
        </li>
        <li>Word Length Statics<br>
            <p>
            Checking word length frequency after data cleaning.<br>
            <img src = "documentation\eda\word_length.png" alt = "Word Length Statics" width = "500">
            </p>
        </li>
        <li>Bigrams and Trigrams<br>
            <p>
            Observing the top 10 bigrams and trigrams.<br>
            <img src = "documentation\eda\bigrams.png" alt = "Top 10 Bigrams" width = "500"><br>
             <img src = "documentation\eda\trigrams.png" alt = "Top 10 Trigrams" width = "500">
            </p>
        </li>
    </ol>

<h1>Model Comparison</h1>
    <p>A simple comparision between bert based sentence cosine similarity and word level cosine similarity is performed.<br>
    when the querry performed is :bag <br>
    <img src = "documentation\model_comparision\bert_sentence_cosine_similarity.png" alt = "Bert Sentence Cosine Similarity" width = "500">
    <img src = "documentation\model_comparision\bert_word_cosine_similarity.png" alt = "Bert Word Cosine Similarity" width = "500">
    </p>

<h1>Model Deployment</h1>
<p>The deployed link for streamlit is <a href =>streamlit.io<a></p>

    