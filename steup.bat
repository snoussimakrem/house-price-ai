@echo off
echo Creating data-warehouse-complete project structure...

mkdir -p data/{raw,processed,external}
mkdir -p notebooks
mkdir -p src/{data,features,models,visualization,api}
mkdir -p tests
mkdir -p model
mkdir -p app/assets
mkdir -p docker
mkdir -p .github/workflows


echo Project structure created successfully.
pause
