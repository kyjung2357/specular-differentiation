import specular


def pytest_report_header(config):
    return f"specular version: {specular.__version__}"