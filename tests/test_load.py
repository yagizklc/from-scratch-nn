from src.main import load_train_data, load_test_data


def test_load_train_data():
    train_data = load_train_data()
    assert len(train_data) == 10
    assert train_data[0][0].exists()
    assert train_data[0][1].exists()
    assert train_data[0][0].stem == train_data[0][1].stem


def test_load_test_data():
    test_data = load_test_data()
    assert len(test_data) == 10
    assert test_data[0][0].exists()
