from setuptools import find_packages, setup

setup(
    name='MarketGym',
    version='0.1.0',
    description='Python Reinforcement Environment for Stock Trading',
    url='https://github.com/jasonpul/MarketGym',
    author='Jason Pul',
    license='MIT',
    packages=['MarketGym'],
    include_package_data=True,
    install_requires=['certifi==2021.5.30', 'charset-normalizer==2.0.4', 'greenlet==1.1.0', 'idna==3.2', 'lxml==4.6.3', 'numpy==1.21.1', 'pandas==1.3.1', 'plotly==5.1.0', 'psycopg2==2.9.1',
                      'python-dateutil==2.8.2', 'python-dotenv==0.19.0', 'pytz==2021.1', 'requests==2.26.0', 'six==1.16.0', 'SQLAlchemy==1.4.22', 'tenacity==8.0.1', 'tqdm==4.62.0', 'urllib3==1.26.6'],
    keywords=['reinforcement', 'stock', 'postgres', 'tradier'],
)
