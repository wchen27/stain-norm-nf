IMAGE ?= wchen27/stain-norm
TAG ?= 0.1.2
PLATFORM ?= linux/amd64

.PHONY: build push

build:
	docker build --provenance=false --platform $(PLATFORM) --tag $(IMAGE):$(TAG) .

push:
	docker push $(IMAGE):$(TAG)
