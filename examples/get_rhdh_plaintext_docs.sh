#!/bin/bash
set -eou pipefail

# Accept multiple versions as arguments
RHDH_VERSIONS=("$@")

# Shared docs repo name (reused each time)
DOCS_REPO_DIR="red-hat-developers-documentation-rhdh"

# Cleanup docs repo on exit
trap "rm -rf $DOCS_REPO_DIR" EXIT

for RHDH_VERSION in "${RHDH_VERSIONS[@]}"; do
    echo "🔄 Processing RHDH version: $RHDH_VERSION"

    # Define paths
    OUTPUT_DIR="rhdh-product-docs-plaintext/${RHDH_VERSION}"
    TOPIC_MAP_REPO_DIR="rhdh-docs-topic-map/${RHDH_VERSION}"

    # Clean previous outputs
    rm -rf "$OUTPUT_DIR" "$DOCS_REPO_DIR" "$TOPIC_MAP_REPO_DIR"

    # Clone RHDH docs repo
    git clone --depth 1 --single-branch --branch "release-${RHDH_VERSION}" \
        https://github.com/redhat-developer/$DOCS_REPO_DIR

    # Clone topic map repo into versioned subfolder
    git clone --depth 1 --single-branch --branch "release-${RHDH_VERSION}" \
        https://github.com/redhat-ai-dev/rhdh-docs-topic-map "$TOPIC_MAP_REPO_DIR"

    # Convert AsciiDoc to plaintext
    python examples/asciidoctor_text/convert_adoc_to_txt_rhdh.py \
        -i "$DOCS_REPO_DIR" \
        -o "$OUTPUT_DIR" \
        -t "$TOPIC_MAP_REPO_DIR/rhdh_topic_map.yaml"

    echo "✅ Finished version $RHDH_VERSION"
    echo "----------------------------------------"
done

echo "🎉 All versions processed successfully."
