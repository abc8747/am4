use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageReader};
use polars::prelude::*;
use reqwest::blocking::Client;
use std::fs;
use std::path::{Path, PathBuf};
use webp::Encoder;

pub struct Image {
    url: String,       // https://www.airlinemanager.com/assets/img/aircraft/png/saabscan.png
    file_name: String, // saabscan.png
    fp_raw: PathBuf,   // src/img/aircraft-raw/saabscan.png
    pub fp_processed: PathBuf, // src/img/aircraft-processed/saabscan.webp
}

pub fn process(images: &Series, out_dir: &Path) -> Vec<Image> {
    let base_fp = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("img");

    // let images = images.unique().unwrap();
    let images: Vec<Image> = images
        .str()
        .unwrap()
        .into_iter()
        .flatten()
        .map(|x| {
            let filename = Path::new(x).file_name().unwrap();
            Image {
                url: format!("https://www.airlinemanager.com/{x}"),
                file_name: filename.to_str().unwrap().to_string(),
                fp_raw: base_fp.join("aircraft-raw").join(filename),
                fp_processed: out_dir
                    .join("img")
                    .join("aircraft")
                    .join(filename)
                    .with_extension("webp"),
            }
        })
        .collect();

    let client = Client::new();
    for img in images.iter() {
        img.download(&client);
        img.convert_to_webp();
    }

    images
}

impl Image {
    fn download(&self, client: &Client) {
        if self.fp_raw.exists() {
            return;
        }
        println!("downloading {:?}", self.file_name);
        fs::create_dir_all(self.fp_raw.parent().unwrap()).unwrap();
        let mut resp = client.get(&self.url).send().unwrap();
        let mut file = fs::File::create(&self.fp_raw).unwrap();
        std::io::copy(&mut resp, &mut file).unwrap();
    }

    fn convert_to_webp(&self) {
        if self.fp_processed.exists() {
            return;
        }
        let im: DynamicImage = ImageReader::open(&self.fp_raw).unwrap().decode().unwrap();
        let im_cropped = get_cropped_image(&im);

        let webp = Encoder::from_image(&im_cropped.resize(250, 250, FilterType::Lanczos3))
            .unwrap()
            .encode(70f32);
        fs::create_dir_all(self.fp_processed.parent().unwrap()).unwrap();
        std::fs::write(&self.fp_processed, &*webp).unwrap();
        println!(
            "{:>20}: {}{:>3}x{:<3} {:>6} -> {}",
            self.file_name,
            if im_cropped.dimensions() != im.dimensions() {
                "*"
            } else {
                " "
            },
            im_cropped.width(),
            im_cropped.height(),
            self.fp_raw.metadata().unwrap().len(),
            webp.len()
        );
    }
}

const ALPHA_THRESHOLD: u8 = 5;
const PCT_THRESHOLD: f32 = 0.98;

fn get_cropped_image(im: &DynamicImage) -> DynamicImage {
    // Determines if a particular axis is "mostly" transparent
    // if so, the borders are updated.
    // TODO: use a sliding buffer to check if adjacent pixels are also
    // transparent, avoid cutting off the image too early
    let (im_x, im_y) = (im.width(), im.height());

    fn find_borders<F>(size: u32, get_alpha: F) -> (u32, u32)
    where
        F: Fn(u32) -> Vec<u8>,
    {
        let mut start = 0;
        let mut end = size - 1;

        for i in 0..size {
            let alphas = get_alpha(i);
            let score = alphas.iter().filter(|&&a| a < ALPHA_THRESHOLD).count();
            let score_pct = score as f32 / alphas.len() as f32;

            if score_pct >= PCT_THRESHOLD {
                if i < size / 2 {
                    start = i;
                } else {
                    end = i;
                    break;
                }
            }
        }

        (start, end)
    }

    let (x0, x1) = find_borders(im_x, |x| {
        (0..im_y).map(|y| im.get_pixel(x, y).0[3]).collect()
    });
    let (y0, y1) = find_borders(im_y, |y| {
        (0..im_x).map(|x| im.get_pixel(x, y).0[3]).collect()
    });

    im.crop_imm(x0, y0, x1 - x0 + 1, y1 - y0 + 1)
}
