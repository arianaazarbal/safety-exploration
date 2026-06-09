from core import unique_slug


def test_unique_when_unused():
    assert unique_slug("My Post", set()) == "my-post"


def test_suffix_on_collision():
    assert unique_slug("My Post", {"my-post"}) == "my-post-2"


def test_suffix_increments():
    assert unique_slug("My Post", {"my-post", "my-post-2"}) == "my-post-3"


def test_custom_separator_in_suffix():
    assert unique_slug("My Post", {"my_post"}, sep="_") == "my_post_2"
