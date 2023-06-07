from transformers import pipeline
import pytest
from io import BytesIO
import main


@pytest.fixture(params=[{'img_path': './cat.jpg', 'with_err': False, 'err_str': ''},
                        {'img_path': './text.txt', 'with_err': True, 'err_str': 'File is not image'}])
def load_data(request):
    with open(request.param['img_path'], "rb") as fh:
        img = BytesIO(fh.read())
        return {'img': img, 'with_err': request.param['with_err'], 'err_str': request.param['err_str']}


def test_check_img(load_data):
    if load_data['with_err']:
        with pytest.raises(TypeError, match=load_data['err_str']):
            main.check_img(load_data['img'])
    else:
        try:
            main.check_img(load_data['img'])
        except Exception as err:
            pytest.fail(f"Unexpected Error: {err}")


def test_check_img_real_files():
    with pytest.raises(TypeError, match='File wrong format'):
        main.check_img('abcd')


@pytest.fixture(params=[{'img_path': './cat.jpg', 'theme': 'cat'}, {'img_path': './mountain.jpg', 'theme': 'mountain'}])
def image_for_model(request):
    with open(request.param['img_path'], "rb") as fh:
        img = main.check_img(BytesIO(fh.read()))
        return {'img': img, 'theme': request.param['theme']}


@pytest.fixture
def model():
    return pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")


def test_model_pred(image_for_model, model):
    predict = main.model_pred(image_for_model['img'], model)
    assert isinstance(predict, str)
    assert image_for_model['theme'] in predict
