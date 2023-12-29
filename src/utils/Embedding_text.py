import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from utils.FS_relief import FS_text_fuction
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
# import umap
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import sent2vec
import utils.consts as consts
from utils.train_neural_text_models import w2vec_encoding,fasttext_encoding,preprocess_sentence
# from text.Autoencoders import SAE_dim_reduction, AE_dim_reduction


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
def pretrained_models (df_data,model,tokenizer):
    encoded_input = tokenizer(df_data, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

def pretrained_models_load(df_data,encoding):
    if encoding == 'mpnet':
        model = SentenceTransformer('all-mpnet-base-v2')
        sentence_embeddings = model.encode(df_data)

    elif encoding == 'Mini':
        model = SentenceTransformer("all-MiniLM-L12-v2")
        sentence_embeddings = model.encode(df_data)

    elif encoding == 'distil':
        model = SentenceTransformer('all-distilroberta-v1')
        sentence_embeddings = model.encode(df_data)

    if encoding == 'multi_mpnet':
        model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        sentence_embeddings = model.encode(df_data)

    elif encoding == 'multi_distil':
        model = SentenceTransformer("multi-qa-distilbert-cos-v1")
        sentence_embeddings = model.encode(df_data)

    elif encoding == 'paraphrase':
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        sentence_embeddings = model.encode(df_data)


    elif encoding == 'Clinical':
        tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        sentence_embeddings=pretrained_models(df_data, model, tokenizer)

    elif encoding == 'Longformer':
        tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer')
        model = AutoModel.from_pretrained('yikuan8/Clinical-Longformer')
        sentence_embeddings = pretrained_models(df_data, model, tokenizer)

    elif encoding == 'SapBert':
        tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
        model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
        sentence_embeddings = pretrained_models(df_data, model, tokenizer)

    elif encoding == 'clinical_fasttext':
        model_path = os.path.join(consts.PATH_PROJECT_DATA_PREPROCESSED_TEXT ,'BioSentVec_PubMed_MIMICIII-bigram_d700.bin')
        model = sent2vec.Sent2vecModel()
        model.load_model(model_path)
        sentences = []
        for j, e in enumerate(df_data):
            u = preprocess_sentence(e)
            o = model.embed_sentence(u)[0]
            sentences.append(o)
        sentence_embeddings = pd.DataFrame(sentences)
    return sentence_embeddings


def Embedded_text(df_data,
                  var_name='raw_medcon',
                  encoding='tfidf',
                  ngrams=1,
                  embedding_size=50,
                  Reduction='None',
                  seed=0,
                  partition=0.2,
                  label='',
                  kernel_KPCA='poly',
                  Feature_selection= False,
                  path=consts.PATH_PROJECT_TEXT_METRICS):

    bow_vectorizer = CountVectorizer(ngram_range=(1, ngrams), max_features=embedding_size)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, ngrams), max_features=embedding_size)

    y_label = label
    df_data = df_data[var_name].tolist()
    if encoding != 'tfidf' and encoding != 'mbow' and encoding != 'w2vec' and encoding != 'fasttext':
        df_data = pretrained_models_load(df_data, encoding)

    list_mae_test = []
    list_mae_train = []
    x_train, x_test, y_train, y_test = train_test_split(pd.DataFrame(df_data), y_label,stratify=y_label, random_state=seed, test_size=partition)

    if encoding == 'tfidf':
        # x_train = pd.DataFrame(x_train)
        # x_test = pd.DataFrame(x_test)
        m_tfidf_sparse = tfidf_vectorizer.fit_transform(x_train.iloc[:,0])
        x_train = m_tfidf_sparse.toarray()
        m_tfidf_sparse = tfidf_vectorizer.transform(x_test.iloc[:,0])
        x_test = m_tfidf_sparse.toarray()

    elif encoding == 'mbow':
        m_bow_sparse = bow_vectorizer.fit_transform(x_train.iloc[:,0])
        x_train = m_bow_sparse.toarray()
        m_bow_sparse = bow_vectorizer.transform(x_test.iloc[:,0])
        x_test = m_bow_sparse.toarray()
    elif encoding == 'w2vec':
        x_train, x_test = w2vec_encoding(x_train.iloc[:,0], x_test.iloc[:,0], embedding_size, seed)

    elif encoding == 'fasttext':
        x_train, x_test = fasttext_encoding(x_train,x_test,ngrams, embedding_size)
    else:
        # pd.DataFrame(sentence_embeddings).to_csv('EMB_TEXT_LATE_TOTAL.csv')
        # pd.DataFrame(x_train).to_csv('EMB_TEXT_LATE_' + str(seed) + '.csv')
        if Reduction == 'PCA':
            mo = PCA(n_components=embedding_size,random_state=0)
            mo = mo.fit(x_train)
            x_train = mo.transform(x_train)
            #pd.DataFrame(x_train).to_csv('PCA_TEXT_LATE_' + str(seed) + '.csv')
            x_test = mo.transform(x_test)
        elif Reduction == 'AE':
            x_train, x_test, mae_train, mae_test = AE_dim_reduction(x_train, x_test, Emb_size=embedding_size)
            list_mae_test.append(mae_test)
            list_mae_train.append(mae_train)

        elif Reduction == 'SAE':
            x_train, x_test, mae_train, mae_test = SAE_dim_reduction(x_train, x_test, Emb_size=embedding_size, )
            list_mae_test.append(mae_test)
            list_mae_train.append(mae_train)

        elif Reduction == 'KPCA':
            mo = KernelPCA(n_components=embedding_size, kernel=kernel_KPCA,random_state=0)
            mo = mo.fit(x_train)
            x_train = mo.transform(x_train)
            x_test = mo.transform(x_test)
        elif Reduction == 'UMAP':
            mo = umap.UMAP(n_neighbors=5,min_dist=0.01,n_components=3,random_state=0)
            mo = mo.fit(x_train, y=y_train)
            x_train = mo.transform(x_train)
            x_test = mo.transform(x_test)
    names = []
    for e in range(embedding_size):
        names.append(str(var_name) + '_' + str(e))
    x_train = pd.DataFrame(x_train, columns=names)
    x_test = pd.DataFrame(x_test, columns=names)

    if Feature_selection!= False:
        features = FS_text_fuction(df_data, var_name, y_label, encoding, Reduction, ngrams, Feature_selection, kernel_KPCA, embedding_size, partition, path)
        return x_train[features], x_test[features], y_train, y_test
    else:
        return x_train, x_test, y_train, y_test


