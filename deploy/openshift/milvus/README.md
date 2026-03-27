
```sh
helm repo add zilliztech https://zilliztech.github.io/milvus-helm/
helm repo update

oc new-project moj-vector-data

oc adm policy add-scc-to-user anyuid -z default 
helm upgrade --install moj -f values.yaml zilliztech/milvus

kubectl port-forward service/moj-milvus 27017:19530

oc expose svc moj-milvus --port 19530 --name moj-api
```

route is listening on /webui/