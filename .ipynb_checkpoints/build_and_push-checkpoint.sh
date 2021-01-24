python3 setup.py sdist bdist_wheel
python3 -m pip install --user --upgrade twine
python3 -m twine upload --repository pypi dist/*
rm -f -r dist/ pyspark_model_plus.egg-info build/
git config --global user.email bhadrarajarshi9@gmail.com
