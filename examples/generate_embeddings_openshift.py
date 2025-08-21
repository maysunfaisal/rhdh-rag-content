#!/usr/bin/env python3
"""Utility script to generate embeddings."""

import logging
import os

from llama_index.readers.file.flat.base import FlatReader

from lightspeed_rag_content import utils
from lightspeed_rag_content.metadata_processor import MetadataProcessor
from lightspeed_rag_content.document_processor import DocumentProcessor

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

OCP_DOCS_ROOT_URL = "https://docs.redhat.com/en/documentation/openshift_container_platform/"
OCP_DOCS_VERSION = "4.15"
RUNBOOKS_ROOT_URL = "https://github.com/openshift/runbooks/blob/master"


class OpenshiftDocsMetadata(MetadataProcessor):
    """Generates metadata from plaintext Openshift documentation."""

    def __init__(self, root_dir: str, ocp_docs_version: str):
        super(OpenshiftDocsMetadata, self).__init__()
        self.root_dir = root_dir
        self.ocp_docs_version = ocp_docs_version

    def url_function(self, file_path: str):

        return (
            OCP_DOCS_ROOT_URL
            + self.ocp_docs_version
            + "/html-single/"
            + file_path.removeprefix(self.root_dir).removesuffix("txt")
            + "/index"
        )


class OpenshiftRunbooksMetadata(MetadataProcessor):

    def __init__(self, root_dir: str):
        super(OpenshiftRunbooksMetadata, self).__init__()
        self.root_dir = root_dir

    def url_function(self, file_path: str):
        return RUNBOOKS_ROOT_URL + file_path.removeprefix(self.root_dir)


if __name__ == "__main__":
    parser = utils.get_common_arg_parser()
    parser.add_argument("-r", "--runbooks", help="Runbooks folder path")
    parser.add_argument(
        "-v", "--ocp-version", help="OCP version", default=OCP_DOCS_VERSION
    )
    args = parser.parse_args()
    print(f"Arguments used: {args}")

    # OLS-823: sanitize directory
    PERSIST_FOLDER = os.path.normpath("/" + args.output).lstrip("/")
    if PERSIST_FOLDER == "":
        PERSIST_FOLDER = "."

    EMBEDDINGS_ROOT_DIR = os.path.abspath(args.folder)
    if EMBEDDINGS_ROOT_DIR.endswith("/"):
        EMBEDDINGS_ROOT_DIR = EMBEDDINGS_ROOT_DIR[:-1]
    RUNBOOKS_ROOT_DIR = os.path.abspath(args.runbooks)
    if RUNBOOKS_ROOT_DIR.endswith("/"):
        RUNBOOKS_ROOT_DIR = RUNBOOKS_ROOT_DIR[:-1]

    # Instantiate Metadata Processor
    print("Instantiate Metadata Processor")
    metadata_processor = OpenshiftDocsMetadata(
        EMBEDDINGS_ROOT_DIR, args.ocp_version)

    runbooks_metadata_processor = OpenshiftRunbooksMetadata(RUNBOOKS_ROOT_DIR)

    # Instantiate Document Processor
    print("Instantiate Document Processor")
    document_processor = DocumentProcessor(
        args.chunk, args.overlap, args.model_name, args.model_dir, args.workers,
        args.vector_store_type, args.index.replace("-", "_"),
    )

    # Process OpenShift documents
    print("Process OpenShift documents")
    document_processor.process(args.folder, metadata=metadata_processor)

    # Process Runbooks
    print("Process Runbooks")
    document_processor.process(
        args.runbooks,
        metadata=runbooks_metadata_processor,
        required_exts=[".md",],
        file_extractor={".md": FlatReader()})

    # Save to the output directory
    document_processor.save(args.index, PERSIST_FOLDER)
