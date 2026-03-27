# General OCP - How Tos


## How to Include ingestion of new Design System




1. create in https://github.com/moj-incub-auth/moj-vector-data organization a repository with the same name as the design system github repo (*IMPORTANT:* make public)

2. clone repo locally

```bash
git clone https://github.com/alphagov/govuk-design-system
cd govuk-design-system
```

3. change remote to `moj-vector-data` org repo and push

```bash
git remote rename origin upstream
git remote add origin git@github.com:moj-incub-auth/govuk-design-system.git
```

4. create automation to trigger ingestion pipeline 

* Create Listener `govuk-design-listener.yaml`

```yaml
apiVersion: triggers.tekton.dev/v1alpha1
kind: EventListener
metadata:
  name: govuk-design-listener
spec:
  triggers:
    - bindings:
        - ref: git-repo-binding
      template:
        ref: govuk-design-template
```

* Create Trigger Template `govuk-design-template.yaml`


```yaml
apiVersion: triggers.tekton.dev/v1alpha1
kind: TriggerTemplate
metadata:
  name: govuk-design-template
spec:
  params:
    - name: gitrevision
    - name: gitrepositoryurl
  resourcetemplates:
    - apiVersion: tekton.dev/v1beta1
      kind: PipelineRun
      metadata:
        generateName: webhook-ingest-govuk-design-piperun-
      spec:
          pipelineRef: 
            name: ingest-content-pipeline
          params:
            # the URL of the GIT repo containing the source data to ingest
            - name: RESOURCE_GIT_URL
              value: "https://github.com/moj-incub-auth/govuk-design-system.git"
            - name: RESOURCE_GIT_BRANCH
              value: "main"
            - name: MILVUS_HOST
              value: "moj-milvus.moj-vector-data.svc.cluster.local"
            - name: RESOURCE_SUBDIRECTORY
              value: "govuk-design-system"
          workspaces:
            - name: shared-data
              volumeClaimTemplate:
                spec:
                  accessModes:
                    - ReadWriteOnce
                  resources:
                    requests:
                      storage: 40Gi
```

* Apply the above either manually or via gitops (*recommended*)

```bash
oc -n moj-builder apply -f govuk-design-listener.yaml
oc -n moj-builder apply -f govuk-design-template.yaml
```

5. Add webhook for automatic trigger

* Get the URL
```bash
oc expose service el-govuk-design-listener -n moj-builder 
oc -n moj-builder  get route  el-govuk-design-listener -o=jsonpath='{.spec.host}'
```

* Create webhook at `https://github.com/moj-incub-auth/govuk-design-system.git` with the ROUTE Host URL 

  * `http://$(oc -n moj-builder  get route  el-govuk-design-listener -o=jsonpath='{.spec.host}')`
  * Content type : `application/json`



