# MOSSNET
Siamese Network for Hand-written Signature Verification [Final Year]

## Setup

We recommend using python3 and a virtual environment.
The default `venv` should be used, or `virtualenv` with `python3`.

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements_cpu.txt
```

If you are using a GPU, you will need to install `tensorflow-gpu` so do:
```bash
pip install -r requirements_gpu.txt
```

## Dataset

Download our [dataset](https://bit.ly/2NwAdPa) and extract it into the directory data/input/.
Any other dataset that may be used shoud be put in the same directory and structured like the one described [here](https://www.kaggle.com/solutionarchitects/mossignatures).

## Training

To run a new training session, do:
```bash
python train.py
```

## Evaluation
To evaluate the performance of the model on new data, do:
```bash
python evaluate.py
```

## Notebook

A [notenook]('final-year-mossnet') is included if it is preferred that the code be run in any python notebook environment. Check out a ran version on [kaggle](https://www.kaggle.com/solutionarchitects/final-year-mossnet)

## Acknowledgement

Our sincere gratitude to oor project supervisor Mr Jephthah Yankey, and the entire computer engineering faculty of the Kwame Nkrumah University of Science and Technology

## Resources

- [Blog post][blog] explaining this project.
- Source code for the built-in TensorFlow function for semi hard online mining triplet loss: [`tf.contrib.losses.metric_learning.triplet_semihard_loss`][tf-triplet-loss].
- [Facenet paper][facenet] introducing online triplet mining
- Detailed explanation of online triplet mining in [*In Defense of the Triplet Loss for Person Re-Identification*][in-defense]
- Blog post by Brandom Amos on online triplet mining: [*OpenFace 0.2.0: Higher accuracy and halved execution time*][openface-blog].
- Source code for the built-in TensorFlow function for semi hard online mining triplet loss: [`tf.contrib.losses.metric_learning.triplet_semihard_loss`][tf-triplet-loss].
- The [coursera lecture][coursera] on triplet loss


[blog]: https://omoindrot.github.io/triplet-loss
[triplet-types-img]: https://omoindrot.github.io/assets/triplet_loss/triplets.png
[triplet-loss-img]: https://omoindrot.github.io/assets/triplet_loss/triplet_loss.png
[online-triplet-loss-img]: https://omoindrot.github.io/assets/triplet_loss/online_triplet_loss.png
[embeddings-img]: https://omoindrot.github.io/assets/triplet_loss/embeddings.png
[embeddings-gif]: https://omoindrot.github.io/assets/triplet_loss/embeddings.gif
[openface-blog]: http://bamos.github.io/2016/01/19/openface-0.2.0/
[facenet]: https://arxiv.org/abs/1503.03832
[in-defense]: https://arxiv.org/abs/1703.07737
[tf-triplet-loss]: https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss
[coursera]: https://www.coursera.org/learn/convolutional-neural-networks/lecture/HuUtN/triplet-loss