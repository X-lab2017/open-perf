from setuptools import setup, find_packages

setup(
    name='openperf',
    version='0.1',
    description='A brief description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='X-lab',
    author_email='fenglin@stu.ecnu.edu.cn',
    url='https://github.com/X-lab2017/open-perf',
    packages=find_packages(),
    package_data={
        'openperf': ['openperf/benchmarks/data_science/bot_detection/data/*.csv'],
    },
    include_package_data=True,
    install_requires=[
        # 这里列出项目的依赖

    ],
    entry_points={
        'console_scripts': [
            'openperf=openperf.main:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
