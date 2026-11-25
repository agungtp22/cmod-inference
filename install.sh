#!/usr/bin/env sh
# cmod-inference one-line installer (macOS ARM64 & Linux x86_64 CPU / NVIDIA CUDA)
# Installs pre-built release packages (.deb on Debian/Ubuntu, .tar.gz on macOS/other Linux)
# with automatic SHA-256 checksum verification — no Rust/source build required.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/agungtp22/cmod-inference/main/install.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/agungtp22/cmod-inference/main/install.sh | CMOD_VERSION=v0.1.0 sh
#   curl -fsSL https://raw.githubusercontent.com/agungtp22/cmod-inference/main/install.sh | CMOD_ACCEL=cuda sh
#   curl -fsSL https://raw.githubusercontent.com/agungtp22/cmod-inference/main/install.sh | CMOD_ACCEL=cpu sh

set -eu

REPO="agungtp22/cmod-inference"
VERSION="${CMOD_VERSION:-latest}"
ACCEL="${CMOD_ACCEL:-auto}" # auto | cuda | rocm | vulkan | cpu

while [ "$#" -gt 0 ]; do
  case "$1" in
    --version|-v)
      VERSION="$2"
      shift 2
      ;;
    --cuda)
      ACCEL="cuda"
      shift
      ;;
    --rocm)
      ACCEL="rocm"
      shift
      ;;
    --vulkan)
      ACCEL="vulkan"
      shift
      ;;
    --cpu)
      ACCEL="cpu"
      shift
      ;;
    --help|-h)
      echo "Usage: install.sh [--version vX.Y.Z] [--cuda | --rocm | --vulkan | --cpu]"
      echo "Env vars: CMOD_VERSION=latest|vX.Y.Z, CMOD_ACCEL=auto|cuda|rocm|vulkan|cpu"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

log() { printf "[cmod-install] %s\n" "$*"; }
warn() { printf "[cmod-install] WARNING: %s\n" "$*" >&2; }
die() { printf "[cmod-install] ERROR: %s\n" "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command '$1' not found."
}

need_cmd curl

# Resolve latest release tag from GitHub API when VERSION=latest
if [ "$VERSION" = "latest" ]; then
  log "Checking latest release for ${REPO}..."
  VERSION=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
    | grep '"tag_name"' | head -n 1 | sed -E 's/.*"tag_name"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')
  [ -n "$VERSION" ] || die "Failed to determine latest release tag from GitHub."
fi

# Strip leading 'v' for Debian version string (e.g. v0.1.0 -> 0.1.0)
VER_NUM="${VERSION#v}"
BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"

OS="$(uname -s)"
ARCH="$(uname -m)"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

verify_sha256() {
  asset_name="$1"
  file_path="$2"
  sums_file="${TMP_DIR}/SHA256SUMS.txt"

  if [ ! -f "$sums_file" ]; then
    if ! curl -fsSL "${BASE_URL}/SHA256SUMS.txt" -o "$sums_file" 2>/dev/null; then
      warn "SHA256SUMS.txt not found for ${VERSION}; skipping checksum verification."
      return 0
    fi
  fi

  expected="$(grep "[[:space:]]${asset_name}\$" "$sums_file" | awk '{print $1}' | head -n 1 || true)"
  if [ -z "$expected" ]; then
    warn "No checksum entry for ${asset_name} in SHA256SUMS.txt."
    return 0
  fi

  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$file_path" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "$file_path" | awk '{print $1}')"
  else
    warn "Neither sha256sum nor shasum found; skipping checksum verification."
    return 0
  fi

  if [ "$actual" != "$expected" ]; then
    die "SHA-256 mismatch for ${asset_name}! Expected ${expected}, got ${actual}."
  fi
  log "SHA-256 verified: ${asset_name}"
}

run_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    die "Root privileges (or sudo) required to run: $*"
  fi
}

case "$OS" in
  Darwin)
    [ "$ARCH" = "arm64" ] || die "macOS pre-built binary supports Apple Silicon (arm64) only (detected: ${ARCH})."
    ASSET="cmod-inference-macos-arm64.tar.gz"
    log "Installing cmod-inference ${VERSION} for macOS ARM64 (Metal)..."
    curl -fL --progress-bar "${BASE_URL}/${ASSET}" -o "${TMP_DIR}/${ASSET}"
    verify_sha256 "$ASSET" "${TMP_DIR}/${ASSET}"

    tar -xzf "${TMP_DIR}/${ASSET}" -C "${TMP_DIR}"
    INSTALL_DIR="${HOME}/.local/bin"
    mkdir -p "$INSTALL_DIR"
    # Always unlink before copying on macOS so AMFI code signature stays valid
    rm -f "${INSTALL_DIR}/cmod-inference"
    cp "${TMP_DIR}/cmod-inference" "${INSTALL_DIR}/cmod-inference"
    chmod +x "${INSTALL_DIR}/cmod-inference"
    xattr -d com.apple.quarantine "${INSTALL_DIR}/cmod-inference" 2>/dev/null || true
    ln -sf "${INSTALL_DIR}/cmod-inference" "${INSTALL_DIR}/cmod"

    log "Installed to ${INSTALL_DIR}/cmod-inference (symlinked as cmod)"
    case ":${PATH}:" in
      *":${INSTALL_DIR}:"*) ;;
      *) warn "Add '${INSTALL_DIR}' to your PATH: export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
    esac
    "${INSTALL_DIR}/cmod" --version
    ;;

  Linux)
    [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ] || die "Linux pre-built package supports x86_64/amd64 only (detected: ${ARCH})."

    # Detect GPU accelerator (NVIDIA CUDA -> AMD ROCm -> Vulkan -> CPU) unless explicitly set
    SELECTED_ACCEL="$ACCEL"
    if [ "$SELECTED_ACCEL" = "auto" ]; then
      if (command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1) || [ -e /dev/nvidia0 ]; then
        SELECTED_ACCEL="cuda"
        log "NVIDIA GPU detected -> selecting CUDA build"
      elif (command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1) || [ -e /dev/kfd ]; then
        SELECTED_ACCEL="rocm"
        log "AMD ROCm GPU detected -> selecting ROCm build"
      elif [ -e /dev/dri/renderD128 ] && (command -v vulkaninfo >/dev/null 2>&1 || [ -f /usr/lib/x86_64-linux-gnu/libvulkan.so.1 ]); then
        SELECTED_ACCEL="vulkan"
        log "Vulkan-capable GPU detected -> selecting Vulkan build"
      else
        SELECTED_ACCEL="cpu"
        log "No supported GPU detected -> selecting CPU build"
      fi
    fi

    pick_linux_asset() {
      pkg_kind="$1" # deb | tar
      wanted="$2"
      if [ "$pkg_kind" = "deb" ]; then
        case "$wanted" in
          cuda)   cand="cmod-inference-cuda_${VER_NUM}_amd64.deb" ;;
          rocm)   cand="cmod-inference-rocm_${VER_NUM}_amd64.deb" ;;
          vulkan) cand="cmod-inference-vulkan_${VER_NUM}_amd64.deb" ;;
          *)      cand="cmod-inference_${VER_NUM}_amd64.deb" ;;
        esac
        fallback="cmod-inference_${VER_NUM}_amd64.deb"
      else
        case "$wanted" in
          cuda)   cand="cmod-inference-linux-x86_64-cuda.tar.gz" ;;
          rocm)   cand="cmod-inference-linux-x86_64-rocm.tar.gz" ;;
          vulkan) cand="cmod-inference-linux-x86_64-vulkan.tar.gz" ;;
          *)      cand="cmod-inference-linux-x86_64.tar.gz" ;;
        esac
        fallback="cmod-inference-linux-x86_64.tar.gz"
      fi

      if [ "$wanted" = "cpu" ]; then
        printf "%s" "$fallback"
        return 0
      fi
      if curl -fsI "${BASE_URL}/${cand}" >/dev/null 2>&1; then
        printf "%s" "$cand"
        return 0
      fi
      if [ "$ACCEL" != "auto" ]; then
        die "Requested ${wanted} package (${cand}) is not available in release ${VERSION}."
      fi
      warn "${cand} not available in release ${VERSION}; falling back to CPU package (${fallback})."
      printf "%s" "$fallback"
    }

    if command -v dpkg >/dev/null 2>&1; then
      # Debian / Ubuntu / Google Colab -> install .deb package
      ASSET="$(pick_linux_asset deb "$SELECTED_ACCEL")"

      log "Downloading ${ASSET} (${VERSION})..."
      curl -fL --progress-bar "${BASE_URL}/${ASSET}" -o "${TMP_DIR}/${ASSET}"
      verify_sha256 "$ASSET" "${TMP_DIR}/${ASSET}"

      log "Installing ${ASSET}..."
      if command -v apt-get >/dev/null 2>&1; then
        run_root apt-get install -y "${TMP_DIR}/${ASSET}"
      else
        run_root dpkg -i "${TMP_DIR}/${ASSET}"
      fi
    else
      # Non-Debian Linux -> install .tar.gz
      ASSET="$(pick_linux_asset tar "$SELECTED_ACCEL")"

      log "Downloading ${ASSET} (${VERSION})..."
      curl -fL --progress-bar "${BASE_URL}/${ASSET}" -o "${TMP_DIR}/${ASSET}"
      verify_sha256 "$ASSET" "${TMP_DIR}/${ASSET}"

      tar -xzf "${TMP_DIR}/${ASSET}" -C "${TMP_DIR}"
      INSTALL_DIR="/usr/local/bin"
      if [ ! -w "$INSTALL_DIR" ] && ! command -v sudo >/dev/null 2>&1; then
        INSTALL_DIR="${HOME}/.local/bin"
        mkdir -p "$INSTALL_DIR"
        install -m 755 "${TMP_DIR}/cmod-inference" "${INSTALL_DIR}/cmod-inference"
        ln -sf "${INSTALL_DIR}/cmod-inference" "${INSTALL_DIR}/cmod"
      else
        run_root install -m 755 "${TMP_DIR}/cmod-inference" "${INSTALL_DIR}/cmod-inference"
        run_root ln -sf "${INSTALL_DIR}/cmod-inference" "${INSTALL_DIR}/cmod"
      fi
    fi

    log "Installation complete!"
    cmod --version
    ;;

  *)
    die "Unsupported OS: ${OS}. For Windows, download cmod-inference-windows-x86_64.zip from ${BASE_URL}."
    ;;
esac
