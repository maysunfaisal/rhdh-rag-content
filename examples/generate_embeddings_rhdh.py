#!/usr/bin/env python3
"""Utility script to generate RHDH embeddings."""

import logging
import os
import yaml

from lightspeed_rag_content import utils
from lightspeed_rag_content.metadata_processor import MetadataProcessor
from lightspeed_rag_content.document_processor import DocumentProcessor

from generate_embeddings_openshift import OCP_DOCS_ROOT_URL, OCP_DOCS_VERSION, OpenshiftDocsMetadata

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

RHDH_DOCS_ROOT_URL = "https://docs.redhat.com/en/documentation/red_hat_developer_hub/"

def process_node(node: dict, dir: str = "", file_url_list: dict = {}) -> dict:
    """Process YAML node from the topic map."""
    currentdir = dir
    if "Topics" in node:
        currentdir = os.path.join(currentdir, node["Dir"])
        for subnode in node["Topics"]:
            file_url_list = process_node(
                subnode, dir=currentdir, file_url_list=file_url_list
            )
    else:
        dir_basename = os.path.basename(currentdir)
        file_url_list[dir_basename] = node["WebpageID"]
    return file_url_list

class RHDHDocsMetadata(MetadataProcessor):
    """Generates metadata from plaintext documentation."""

    def __init__(self, root_dir: str, rhdh_docs_version: str, topic_map_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self.rhdh_docs_version = rhdh_docs_version
        self.topic_map_dir = topic_map_dir
        self.file_url_list: dict = {}

        topic_map_file = os.path.join(topic_map_dir, rhdh_docs_version, "rhdh_topic_map.yaml")
        if not os.path.isfile(topic_map_file):
            raise FileNotFoundError(f"Topic map not found at: {topic_map_file}")

        with open(topic_map_file, "r") as fin:
            topic_map_data = yaml.safe_load_all(fin)
            for map_node in topic_map_data:
                self.file_url_list = process_node(map_node, file_url_list=self.file_url_list)

    def url_function(self, file_path: str):
        dir_basename = os.path.basename(os.path.dirname(file_path))
        return (
            RHDH_DOCS_ROOT_URL
            + self.rhdh_docs_version
            + "/html-single/"
            + self.file_url_list.get(dir_basename, "unknown")
            + "/index"
        )


if __name__ == "__main__":
    parser = utils.get_common_arg_parser()
    parser.add_argument("--topic-map", "-t", required=True, help="The topic map directory")
    parser.add_argument(
        "-v_ocp", "--ocp-version", help="OCP version", default=OCP_DOCS_VERSION
    )
    args = parser.parse_args()
    print(f"Arguments used: {args}")

    topic_map_dir = os.path.normpath(os.path.join(os.getcwd(), args.topic_map))

    # OLS-823: sanitize directory
    PERSIST_FOLDER = os.path.normpath("/" + args.output).lstrip("/")
    if PERSIST_FOLDER == "":
        PERSIST_FOLDER = "."

    EMBEDDINGS_ROOT_DIR_RHDH = os.path.abspath(args.folder_rhdh)
    if EMBEDDINGS_ROOT_DIR_RHDH.endswith("/"):
        EMBEDDINGS_ROOT_DIR_RHDH = EMBEDDINGS_ROOT_DIR_RHDH[:-1]
        
    EMBEDDINGS_ROOT_DIR_OCP = os.path.abspath(args.folder_ocp)
    if EMBEDDINGS_ROOT_DIR_OCP.endswith("/"):
        EMBEDDINGS_ROOT_DIR_OCP = EMBEDDINGS_ROOT_DIR_OCP[:-1]

    # Instantiate Document Processor
    print("Instantiate Document Processor")
    document_processor = DocumentProcessor(
        args.chunk, args.overlap, args.model_name, args.model_dir, args.workers,
        args.vector_store_type, args.index.replace("-", "_"),
    )

    # Auto-discover versions from the folder (subdirectories)
    rhdh_versions = sorted([
        d for d in os.listdir(EMBEDDINGS_ROOT_DIR_RHDH)
        if os.path.isdir(os.path.join(EMBEDDINGS_ROOT_DIR_RHDH, d))
    ])

    if not rhdh_versions:
        raise RuntimeError(f"No RHDH version folders found in {EMBEDDINGS_ROOT_DIR_RHDH}")

    for version in rhdh_versions:
        version_docs_path = os.path.join(EMBEDDINGS_ROOT_DIR_RHDH, version)
        print(f"Processing RHDH {version} from {version_docs_path}")

        # Create a metadata processor for this RHDH version
        metadata_processor_rhdh = RHDHDocsMetadata(version_docs_path, version, topic_map_dir)

        # Process this RHDH version's docs
        document_processor.process(
            docs_dir=version_docs_path,
            metadata=metadata_processor_rhdh,
            version=version,
        )
    
    # Create a metadata processor for this OCP version
    metadata_processor_ocp = OpenshiftDocsMetadata(
        EMBEDDINGS_ROOT_DIR_OCP, args.ocp_version)
    
    # Process OpenShift documents
    print("Process OpenShift documents")
    document_processor.process(docs_dir=args.folder_ocp, metadata=metadata_processor_ocp, version=args.ocp_version)

    # Save to the output directory
    print("Saving index...")
    document_processor.save(args.index, PERSIST_FOLDER)
