# Comparative Analysis of Fashion Captioning for Multimodal Fashion Recommendation


## Image captioning

To run the evaluation you first need to run
```
chmod +x get_stanford_models.sh 
./get_stanford_models.sh
```
which will unpack the necessary Stanford CoreNLP packages into the spice package.

Missing paraphrase file for the meteor package can be found [here](https://github.com/tylin/coco-caption/blob/3a9afb2682141a03e1cdc02b0df6770d2c884f6f/pycocoevalcap/meteor/data/paraphrase-en.gz).

### Preprocessing H&M
To preprocess the item descriptions to extract the attributes of items (based on [Yang et al.](https://arxiv.org/pdf/2008.02693.pdf)), we used [Stanza](https://stanfordnlp.github.io/stanza/) an updated version from the original Stanford Parser.
