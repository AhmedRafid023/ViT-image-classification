#!/usr/bin/env bash
set -euo pipefail

ALL_DATASETS=("caltech-101" "oxford_pets" "stanford_cars" "oxford_flowers" "food-101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101" "imagenet" "imagenetv2" "imagenet-sketch" "imagenet-adversarial" "imagenet-rendition")

usage() {
  cat <<'EOF'
Usage:
  DATA=/path ./download_data.sh --all
  ./download_data.sh --data /path --individual caltech-101,oxford_pets
  ./download_data.sh --data /path --all_except oxford_pets

Datasets:
  caltech-101, oxford_pets, stanford_cars, oxford_flowers, food-101, fgvc_aircraft, sun397, dtd, eurosat, ucf101, imagenet, imagenetv2, imagenet-sketch, imagenet-adversarial, imagenet-rendition
EOF
}

contains_dataset() {
  local target="$1"
  shift
  local item
  for item in "$@"; do
    [[ "$item" == "$target" ]] && return 0
  done
  return 1
}

split_csv() {
  local csv="$1"
  local -n out_arr_ref="$2"
  out_arr_ref=()
  IFS=',' read -r -a out_arr_ref <<< "$csv"
}

download_caltech_101() {
  local data_root="$1"
  local dir="${data_root}/caltech-101"
  local zip_path="${dir}/caltech-101.zip"
  local zip_url="https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
  local split_json="${dir}/split_zhou_Caltech101.json"
  local split_url="https://drive.google.com/uc?export=download&id=1hyarUivQE36mY6jSomru6Fjd-JzwcCzN"

  echo "[caltech-101] Preparing ${dir}"
  mkdir -p "${dir}"
  curl -fL "${zip_url}" -o "${zip_path}"
  unzip -o "${zip_path}" -d "${dir}"
  tar -xzf "${dir}/caltech-101/101_ObjectCategories.tar.gz" -C "${dir}"
  curl -fL "${split_url}" -o "${split_json}"

  rm -f "${zip_path}"
  rm -rf "${dir}/caltech-101" "${dir}/__MACOSX"
  rm -f "${dir}/Annotations.tar" "${dir}/show_annotation.m"
}

download_oxford_pets() {
  local data_root="$1"
  local dir="${data_root}/oxford_pets"
  local images_tar="${dir}/images.tar.gz"
  local images_url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
  local ann_tar="${dir}/annotations.tar.gz"
  local ann_url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
  local split_json="${dir}/split_zhou_OxfordPets.json"
  local split_url="https://drive.google.com/uc?export=download&id=1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs"

  echo "[oxford_pets] Preparing ${dir}"
  mkdir -p "${dir}"
  curl -fL "${images_url}" -o "${images_tar}"
  curl -fL "${ann_url}" -o "${ann_tar}"
  tar -xzf "${images_tar}" -C "${dir}"
  tar -xzf "${ann_tar}" -C "${dir}"
  curl -fL "${split_url}" -o "${split_json}"

  rm -f "${images_tar}" "${ann_tar}"
}

download_stanford_cars() {
  local data_root="$1"
  local dir="${data_root}/stanford_cars"
  local source_repo="https://github.com/jhpohovey/StanfordCars-Dataset.git"
  local tmp_clone_dir="${data_root}/.tmp_stanfordcars_clone"
  local split_json="${dir}/split_zhou_StanfordCars.json"
  local split_url="https://drive.google.com/uc?export=download&id=1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT"

  echo "[stanford_cars] Preparing ${dir}"
  mkdir -p "${dir}"

  if [[ ! -d "${dir}/cars_train" || ! -d "${dir}/cars_test" || ! -d "${dir}/devkit" || ! -f "${dir}/cars_test_annos_withlabels.mat" ]]; then
    rm -rf "${tmp_clone_dir}"
    git clone --depth 1 "${source_repo}" "${tmp_clone_dir}"

    rm -rf "${dir}/cars_train" "${dir}/cars_test" "${dir}/devkit" "${dir}/cars_test_annos_withlabels.mat"
    cp -r "${tmp_clone_dir}/stanford_cars/cars_train" "${dir}/"
    cp -r "${tmp_clone_dir}/stanford_cars/cars_test" "${dir}/"
    cp -r "${tmp_clone_dir}/stanford_cars/devkit" "${dir}/"
    cp "${tmp_clone_dir}/stanford_cars/cars_test_annos_withlabels.mat" "${dir}/"
    rm -rf "${tmp_clone_dir}"
  else
    echo "[stanford_cars] Core files already present, skipping clone."
  fi

  curl -fL "${split_url}" -o "${split_json}"
}

download_oxford_flowers() {
  local data_root="$1"
  local dir="${data_root}/oxford_flowers"
  local images_tgz="${dir}/102flowers.tgz"
  local images_url="https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
  local labels_mat="${dir}/imagelabels.mat"
  local labels_url="https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
  local cat_to_name_json="${dir}/cat_to_name.json"
  local cat_to_name_url="https://drive.google.com/uc?export=download&id=1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0"
  local split_json="${dir}/split_zhou_OxfordFlowers.json"
  local split_url="https://drive.google.com/uc?export=download&id=1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT"

  echo "[oxford_flowers] Preparing ${dir}"
  mkdir -p "${dir}"

  curl -fL "${images_url}" -o "${images_tgz}"
  curl -fL "${labels_url}" -o "${labels_mat}"
  curl -fL "${cat_to_name_url}" -o "${cat_to_name_json}"
  curl -fL "${split_url}" -o "${split_json}"

  tar -xzf "${images_tgz}" -C "${dir}"
  rm -f "${images_tgz}"
}

download_food_101() {
  local data_root="$1"
  local archive_path="${data_root}/food-101.tar.gz"
  local archive_url="http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
  local dir="${data_root}/food-101"
  local split_json="${dir}/split_zhou_Food101.json"
  local split_url="https://drive.google.com/uc?export=download&id=1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl"

  echo "[food-101] Preparing ${dir}"
  mkdir -p "${data_root}"

  curl -fL "${archive_url}" -o "${archive_path}"
  tar -xzf "${archive_path}" -C "${data_root}"
  curl -fL "${split_url}" -o "${split_json}"

  rm -f "${archive_path}"
}

download_fgvc_aircraft() {
  local data_root="$1"
  local archive_path="${data_root}/fgvc-aircraft-2013b.tar.gz"
  local archive_url="https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
  local extracted_root="${data_root}/fgvc-aircraft-2013b"
  local final_dir="${data_root}/fgvc_aircraft"

  echo "[fgvc_aircraft] Preparing ${final_dir}"
  mkdir -p "${data_root}"

  curl -fL "${archive_url}" -o "${archive_path}"
  tar -xzf "${archive_path}" -C "${data_root}"

  rm -rf "${final_dir}"
  mv "${extracted_root}/data" "${final_dir}"

  rm -rf "${extracted_root}"
  rm -f "${archive_path}"
}

download_sun397() {
  local data_root="$1"
  local dir="${data_root}/sun397"
  local partitions_zip="${dir}/Partitions.zip"
  local partitions_url="https://3dvision.princeton.edu/projects/2010/SUN/download/Partitions.zip"
  local split_json="${dir}/split_zhou_SUN397.json"
  local split_url="https://drive.google.com/uc?export=download&id=1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq"

  echo "[sun397] Preparing ${dir}"
  mkdir -p "${dir}/SUN397"

  echo "[sun397] Downloading Partitions.zip for original filenames ..."
  curl -fL "${partitions_url}" -o "${partitions_zip}"
  unzip -o "${partitions_zip}" -d "${dir}"

  echo "[sun397] Downloading images from HuggingFace (1aurent/SUN397, ~39 GB) ..."
  python3 - <<PYEOF
import os, sys, zipfile
from pathlib import Path
from collections import defaultdict

# Remove cwd from sys.path so the local datasets/ folder does not shadow
# the installed HuggingFace datasets package.
_cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ("", ".", _cwd)]

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "Pillow"])
    sys.path = [p for p in sys.path if p not in ("", ".", _cwd)]
    from datasets import load_dataset

sun397_dir = Path("${dir}/SUN397")
partitions_zip = Path("${dir}/Partitions.zip")
sun397_dir.mkdir(parents=True, exist_ok=True)

# The HuggingFace dataset was built with sorted(rglob("*.jpg")), which is a
# global lexicographic sort. Because class paths prefix the filename, this is
# equivalent to: per class, images appear in alphabetical filename order.
# Partitions.zip lists every original sun_<hash>.jpg per class; sorting those
# names alphabetically gives the same order used during HuggingFace conversion,
# so we can assign original filenames by class-local index.
print("[sun397] Parsing Partitions.zip for original filenames ...", flush=True)
filenames_per_class = defaultdict(set)
with zipfile.ZipFile(str(partitions_zip)) as z:
    for zname in z.namelist():
        if not zname.endswith(".txt"):
            continue
        with z.open(zname) as f:
            for line in f:
                path = line.decode().strip()
                if not path:
                    continue
                # Strip known prefixes like SUN397/ or data/SUN397/
                parts = path.lstrip("/").split("/")
                while parts and parts[0] in ("data", "SUN397"):
                    parts = parts[1:]
                if len(parts) < 2:
                    continue
                class_key = "/" + "/".join(parts[:-1])
                filenames_per_class[class_key].add(parts[-1])

sorted_filenames = {cls: sorted(fnames) for cls, fnames in filenames_per_class.items()}
class_counters = defaultdict(int)

print("[sun397] Loading dataset from HuggingFace ...", flush=True)
ds = load_dataset("1aurent/SUN397", split="train")
total = len(ds)
id2label = ds.features["label"].int2str  # callable: int -> "/a/abbey"

for i, item in enumerate(ds):
    label = id2label(item["label"])  # e.g. /a/abbey
    idx = class_counters[label]
    class_counters[label] += 1

    fnames = sorted_filenames.get(label, [])
    fname = fnames[idx] if idx < len(fnames) else f"sun397_{i:06d}.jpg"

    rel = label.lstrip("/").replace("/", os.sep)
    label_dir = sun397_dir / rel
    label_dir.mkdir(parents=True, exist_ok=True)

    img_path = label_dir / fname
    if not img_path.exists():
        item["image"].save(img_path, "JPEG")

    if (i + 1) % 5000 == 0:
        print(f"[sun397] {i + 1}/{total} images saved ...", flush=True)

print(f"[sun397] All {total} images saved.", flush=True)
PYEOF

  curl -fL "${split_url}" -o "${split_json}"

  rm -f "${partitions_zip}" "${dir}/split10.mat"
}

download_dtd() {
  local data_root="$1"
  local archive_path="${data_root}/dtd-r1.0.1.tar.gz"
  local archive_url="https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
  local dir="${data_root}/dtd"
  local split_json="${dir}/split_zhou_DescribableTextures.json"
  local split_url="https://drive.google.com/uc?export=download&id=1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x"

  echo "[dtd] Preparing ${dir}"
  mkdir -p "${data_root}"

  curl -fL "${archive_url}" -o "${archive_path}"
  tar -xzf "${archive_path}" -C "${data_root}"
  curl -fL "${split_url}" -o "${split_json}"

  rm -f "${archive_path}"
}

download_eurosat() {
  local data_root="$1"
  local dir="${data_root}/eurosat"
  local archive_path="${dir}/EuroSAT_RGB.zip"
  local archive_url="https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
  local split_json="${dir}/split_zhou_EuroSAT.json"
  local split_url="https://drive.google.com/uc?export=download&id=1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o"

  echo "[eurosat] Preparing ${dir}"
  mkdir -p "${dir}"

  curl -fL "${archive_url}" -o "${archive_path}"
  unzip -o "${archive_path}" -d "${dir}"
  curl -fL "${split_url}" -o "${split_json}"

  # Enforce final structure:
  # eurosat/2750/
  # eurosat/split_zhou_EuroSAT.json
  # EuroSAT zip typically extracts as EuroSAT_RGB/<class_folders>.
  if [[ -d "${dir}/EuroSAT_RGB" ]]; then
    rm -rf "${dir}/2750"
    mv "${dir}/EuroSAT_RGB" "${dir}/2750"
  elif [[ -d "${dir}/2750" ]]; then
    :
  else
    echo "[eurosat] ERROR: expected EuroSAT_RGB or 2750 directory not found after extraction."
    exit 1
  fi

  rm -f "${archive_path}"
}

download_ucf101() {
  local data_root="$1"
  local dir="${data_root}/ucf101"
  local archive_path="${dir}/UCF-101-midframes.zip"
  local archive_url="https://drive.usercontent.google.com/download?id=10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O&export=download&confirm=t"
  local split_json="${dir}/split_zhou_UCF101.json"
  local split_url="https://drive.google.com/uc?export=download&id=1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y"

  echo "[ucf101] Preparing ${dir}"
  mkdir -p "${dir}"

  curl -fL "${archive_url}" -o "${archive_path}"
  if ! unzip -tq "${archive_path}" > /dev/null 2>&1; then
    echo "[ucf101] ERROR: downloaded file is not a valid zip archive."
    echo "[ucf101] This usually means Google Drive returned an HTML page instead of the file."
    echo "[ucf101] Try again later or download manually into ${archive_path} and rerun."
    exit 1
  fi
  unzip -o "${archive_path}" -d "${dir}"
  curl -fL "${split_url}" -o "${split_json}"

  if [[ ! -d "${dir}/UCF-101-midframes" ]]; then
    echo "[ucf101] ERROR: expected UCF-101-midframes directory not found after extraction."
    exit 1
  fi

  rm -f "${archive_path}"
}

download_imagenet() {
  local data_root="$1"
  local dir="${data_root}/imagenet"
  local images_dir="${dir}/images"
  local train_tar="${images_dir}/train_blurred.tar.gz"
  local val_tar="${images_dir}/val_blurred.tar.gz"
  local train_url="https://image-net.org/data/ILSVRC/blurred/train_blurred.tar.gz"
  local val_url="https://image-net.org/data/ILSVRC/blurred/val_blurred.tar.gz"
  local classnames_dst="${dir}/classnames.txt"
  local classnames_url="https://drive.google.com/uc?export=download&id=1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF"

  echo "[imagenet] Preparing ${images_dir}"
  mkdir -p "${images_dir}"

  curl -fL "${train_url}" -o "${train_tar}"
  tar -xzf "${train_tar}" -C "${images_dir}"
  if [[ -d "${images_dir}/train_blurred" ]]; then
    mv "${images_dir}/train_blurred" "${images_dir}/train"
  elif [[ ! -d "${images_dir}/train" ]]; then
    echo "[imagenet] ERROR: expected train or train_blurred directory not found after extraction."
    exit 1
  fi

  curl -fL "${val_url}" -o "${val_tar}"
  tar -xzf "${val_tar}" -C "${images_dir}"
  if [[ -d "${images_dir}/val_blurred" ]]; then
    mv "${images_dir}/val_blurred" "${images_dir}/val"
  elif [[ ! -d "${images_dir}/val" ]]; then
    echo "[imagenet] ERROR: expected val or val_blurred directory not found after extraction."
    exit 1
  fi

  curl -fL "${classnames_url}" -o "${classnames_dst}"

  rm -f "${train_tar}" "${val_tar}"
}

download_imagenetv2() {
  local data_root="$1"
  local dir="${data_root}/imagenetv2"
  local archive_path="${dir}/imagenetv2-matched-frequency.tar.gz"
  local archive_url="https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz?download=true"
  local classnames_dst="${dir}/classnames.txt"
  local classnames_url="https://drive.google.com/uc?export=download&id=1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF"

  echo "[imagenetv2] Preparing ${dir}"
  mkdir -p "${dir}"

  curl -fL "${archive_url}" -o "${archive_path}"
  tar -xf "${archive_path}" -C "${dir}"
  curl -fL "${classnames_url}" -o "${classnames_dst}"

  rm -f "${archive_path}"
}

download_imagenet_sketch() {
  local data_root="$1"
  local dir="${data_root}/imagenet-sketch"
  local archive_path="${dir}/ImageNet-Sketch.zip"
  local archive_url="https://drive.usercontent.google.com/download?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA&export=download&confirm=t"
  local extracted_root_repo="${dir}/ImageNet-Sketch-master"
  local extracted_root_data="${dir}/ImageNet-Sketch"
  local classnames_dst="${dir}/classnames.txt"
  local classnames_url="https://drive.google.com/uc?export=download&id=1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF"

  echo "[imagenet-sketch] Preparing ${dir}"
  mkdir -p "${dir}"

  curl -fL "${archive_url}" -o "${archive_path}"
  unzip -o "${archive_path}" -d "${dir}"

  if [[ -d "${extracted_root_repo}/sketch" ]]; then
    rm -rf "${dir}/images"
    mv "${extracted_root_repo}/sketch" "${dir}/images"
  elif [[ -d "${extracted_root_repo}/images" ]]; then
    rm -rf "${dir}/images"
    mv "${extracted_root_repo}/images" "${dir}/images"
  elif [[ -d "${extracted_root_data}/sketch" ]]; then
    rm -rf "${dir}/images"
    mv "${extracted_root_data}/sketch" "${dir}/images"
  elif [[ -d "${extracted_root_data}/images" ]]; then
    rm -rf "${dir}/images"
    mv "${extracted_root_data}/images" "${dir}/images"
  elif [[ -d "${dir}/sketch" ]]; then
    rm -rf "${dir}/images"
    mv "${dir}/sketch" "${dir}/images"
  elif [[ -d "${dir}/images" ]]; then
    :
  else
    echo "[imagenet-sketch] ERROR: expected sketch/images directory not found after extraction."
    exit 1
  fi

  curl -fL "${classnames_url}" -o "${classnames_dst}"

  rm -rf "${extracted_root_repo}" "${extracted_root_data}"
  rm -f "${archive_path}"
}

download_imagenet_adversarial() {
  local data_root="$1"
  local dir="${data_root}/imagenet-adversarial"
  local archive_path="${dir}/imagenet-a.tar"
  local archive_url="https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
  local classnames_dst="${dir}/classnames.txt"
  local classnames_url="https://drive.google.com/uc?export=download&id=1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF"

  echo "[imagenet-adversarial] Preparing ${dir}"
  mkdir -p "${dir}"

  curl -fL "${archive_url}" -o "${archive_path}"
  tar -xf "${archive_path}" -C "${dir}"

  if [[ ! -d "${dir}/imagenet-a" ]]; then
    echo "[imagenet-adversarial] ERROR: expected imagenet-a directory not found after extraction."
    exit 1
  fi

  curl -fL "${classnames_url}" -o "${classnames_dst}"

  rm -f "${archive_path}"
}

download_imagenet_rendition() {
  local data_root="$1"
  local dir="${data_root}/imagenet-rendition"
  local archive_path="${dir}/imagenet-r.tar"
  local archive_url="https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
  local classnames_dst="${dir}/classnames.txt"
  local classnames_url="https://drive.google.com/uc?export=download&id=1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF"

  echo "[imagenet-rendition] Preparing ${dir}"
  mkdir -p "${dir}"

  curl -fL "${archive_url}" -o "${archive_path}"
  tar -xf "${archive_path}" -C "${dir}"

  if [[ ! -d "${dir}/imagenet-r" ]]; then
    echo "[imagenet-rendition] ERROR: expected imagenet-r directory not found after extraction."
    exit 1
  fi

  curl -fL "${classnames_url}" -o "${classnames_dst}"

  rm -f "${archive_path}"
}

DATA_ROOT="${DATA:-}"
MODE=""
DATASET_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)
      DATA_ROOT="$2"
      shift 2
      ;;
    --all)
      MODE="all"
      shift
      ;;
    --individual)
      MODE="individual"
      DATASET_ARG="$2"
      shift 2
      ;;
    --all_except)
      MODE="all_except"
      DATASET_ARG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${DATA_ROOT}" || -z "${MODE}" ]]; then
  usage
  exit 1
fi

SELECTED=()
case "${MODE}" in
  all)
    SELECTED=("${ALL_DATASETS[@]}")
    ;;
  individual)
    split_csv "${DATASET_ARG}" SELECTED
    ;;
  all_except)
    EXCLUDED=()
    split_csv "${DATASET_ARG}" EXCLUDED
    for d in "${ALL_DATASETS[@]}"; do
      if ! contains_dataset "$d" "${EXCLUDED[@]}"; then
        SELECTED+=("$d")
      fi
    done
    ;;
  *)
    echo "Invalid mode: ${MODE}"
    exit 1
    ;;
esac

for d in "${SELECTED[@]}"; do
  case "$d" in
    caltech-101)
      download_caltech_101 "${DATA_ROOT}"
      ;;
    oxford_pets)
      download_oxford_pets "${DATA_ROOT}"
      ;;
    stanford_cars)
      download_stanford_cars "${DATA_ROOT}"
      ;;
    oxford_flowers)
      download_oxford_flowers "${DATA_ROOT}"
      ;;
    food-101)
      download_food_101 "${DATA_ROOT}"
      ;;
    fgvc_aircraft)
      download_fgvc_aircraft "${DATA_ROOT}"
      ;;
    sun397)
      download_sun397 "${DATA_ROOT}"
      ;;
    dtd)
      download_dtd "${DATA_ROOT}"
      ;;
    eurosat)
      download_eurosat "${DATA_ROOT}"
      ;;
    ucf101)
      download_ucf101 "${DATA_ROOT}"
      ;;
    imagenet)
      download_imagenet "${DATA_ROOT}"
      ;;
    imagenetv2)
      download_imagenetv2 "${DATA_ROOT}"
      ;;
    imagenet-sketch)
      download_imagenet_sketch "${DATA_ROOT}"
      ;;
    imagenet-adversarial)
      download_imagenet_adversarial "${DATA_ROOT}"
      ;;
    imagenet-rendition)
      download_imagenet_rendition "${DATA_ROOT}"
      ;;
    *)
      echo "Unsupported dataset: ${d}"
      echo "Supported: ${ALL_DATASETS[*]}"
      exit 1
      ;;
  esac
done

echo "Done."
