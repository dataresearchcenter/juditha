from unittest.mock import MagicMock, patch

from juditha import predict


def test_sample_aggregator(store, eu_authorities):
    """Test SampleAggregator class"""
    # Load test data into aggregator
    for entity in eu_authorities:
        store.aggregator.put(entity)
    store.aggregator.flush()

    # Create SampleAggregator and make sample
    sampler = predict.SampleAggregator(store.aggregator, limit=10)
    sampler.make_sample()

    # Check we collected data
    assert sampler.collected > 0
    assert len(sampler.names) > 0

    # Check sample format from iterate
    samples = list(sampler.iterate())
    assert len(samples) > 0

    for label, text in samples:
        assert label.startswith("__label__")
        assert isinstance(text, str)
        assert len(text) > 0


@patch("random.choice")
@patch("random.randint")
def test_create_training_data(mock_randint, mock_choice, store, eu_authorities):
    """Test training data file creation with noise"""
    # Load test data
    for entity in eu_authorities:
        store.aggregator.put(entity)
    store.aggregator.flush()

    # Mock noise generation to be predictable
    mock_choice.return_value = "char_add"
    mock_randint.return_value = 0

    # Create SampleAggregator and training data
    sampler = predict.SampleAggregator(store.aggregator, limit=20, train_ratio=0.8)
    sampler.make_sample()
    train_file, val_file = sampler.create_training_data()

    try:
        # Check files exist
        assert train_file.exists()
        assert val_file.exists()

        # Check file contents
        train_lines = train_file.read_text().strip().split("\n")
        val_lines = val_file.read_text().strip().split("\n")

        assert len(train_lines) > 0
        assert len(val_lines) > 0

        # Check format
        for line in train_lines[:3]:  # Check first few lines
            if line.strip():  # Skip empty lines
                parts = line.split(" ", 1)
                assert len(parts) == 2
                assert parts[0].startswith("__label__")

        for line in val_lines[:3]:  # Check first few lines
            if line.strip():  # Skip empty lines
                parts = line.split(" ", 1)
                assert len(parts) == 2
                assert parts[0].startswith("__label__")

    finally:
        # Cleanup
        train_file.unlink(missing_ok=True)
        val_file.unlink(missing_ok=True)


@patch("fasttext.load_model")
@patch("fasttext.train_supervised")
def test_train_model(
    mock_train_supervised, _mock_load_model, store, eu_authorities, tmp_path
):
    """Test model training"""
    # Setup mocks
    mock_model = MagicMock()
    mock_model.test.return_value = (None, 0.85)  # (precision, recall)
    mock_train_supervised.return_value = mock_model

    # Load test data
    for entity in eu_authorities:
        store.aggregator.put(entity)
    store.aggregator.flush()

    model_path = str(tmp_path / "test_model.bin")

    # Train model
    predict.train_model(store.aggregator, model_path=model_path, limit=20, verbose=0)

    # Check training was called with correct parameters
    mock_train_supervised.assert_called_once()
    _, kwargs = mock_train_supervised.call_args
    assert "input" in kwargs
    assert kwargs["epoch"] == 25  # default value
    assert kwargs["lr"] == 0.1  # default value

    # Check model was saved
    mock_model.save_model.assert_called_once_with(model_path)


@patch("fasttext.load_model")
def test_predict_schema(mock_load_model, tmp_path):
    """Test schema prediction"""
    # Setup mock
    mock_model = MagicMock()
    mock_model.predict.return_value = (
        ["__label__Person", "__label__Organization", "__label__Company"],
        [0.95, 0.03, 0.02],
    )
    mock_load_model.return_value = mock_model

    model_path = str(tmp_path / "test_model.bin")

    # Test prediction
    results = list(predict.predict_schema("John Doe", model_path=model_path))

    assert len(results) == 1
    assert results[0].label == "Person"
    assert results[0].score == 0.95

    # Check model was loaded and called correctly
    mock_load_model.assert_called_once_with(model_path)
    mock_model.predict.assert_called_once_with("john doe", k=3)


@patch("fasttext.load_model")
def test_predict_schema_no_predictions(mock_load_model):
    """Test schema prediction when no predictions are returned"""
    # Setup mock with empty predictions
    mock_model = MagicMock()
    mock_model.predict.return_value = ([], [])
    mock_load_model.return_value = mock_model

    results = list(predict.predict_schema("", model_path="dummy.bin"))
    assert results == []


def test_default_normalize():
    """Test the default normalization function"""
    assert predict.default_normalize("John Doe") == "john doe"
    assert predict.default_normalize("COMPANY INC.") == "company inc"
    assert predict.default_normalize("  Mixed Case  ") == "mixed case"


@patch("random.choice")
@patch("random.randint")
def test_add_noise_char_swap(mock_randint, mock_choice):
    """Test character swap noise"""
    mock_choice.return_value = "char_swap"
    mock_randint.return_value = 1  # Swap positions 1 and 2

    result = predict.add_noise("john")
    assert result == "jhon"  # positions 1 and 2 swapped ('o' and 'h')


@patch("random.choice")
@patch("random.randint")
def test_add_noise_char_drop(mock_randint, mock_choice):
    """Test character drop noise"""
    mock_choice.return_value = "char_drop"
    mock_randint.return_value = 1  # Drop character at position 1

    result = predict.add_noise("john")
    assert result == "jhn"  # 'o' dropped


@patch("random.choice")
@patch("random.randint")
def test_add_noise_char_add(mock_randint, mock_choice):
    """Test character add noise"""
    mock_choice.side_effect = ["char_add", "x"]  # Add 'x' at position
    mock_randint.return_value = 1  # Add at position 1

    result = predict.add_noise("john")
    assert result == "jxohn"  # 'x' added at position 1


@patch("random.choice")
@patch("random.randint")
def test_add_noise_word_duplicate(mock_randint, mock_choice):
    """Test word duplication noise"""
    mock_choice.side_effect = ["word_duplicate", "john"]  # Duplicate "john"
    mock_randint.return_value = 1  # Insert at position 1

    result = predict.add_noise("john doe")
    assert result == "john john doe"  # "john" duplicated


def test_add_noise_short_text():
    """Test that very short text is returned unchanged"""
    result = predict.add_noise("jo")
    assert result == "jo"  # Should return unchanged for text < 3 chars
