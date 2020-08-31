use om_sd::na;
use na::{Vector2, Vector3};
use std::fs::File;
use std::io::Write;

fn steepest_descent_with_both() {
    let f    = |x: Vector2<f64>| x[0].powi(2) + x.norm_squared().exp() + 4.0 * x[0] + 3.0 * x[1];
    let grad = |x: Vector2<f64>| Vector2::new(
        2.0 * x[0] * (1.0 + x.norm_squared().exp()) + 4.0,
        2.0 * x[1] * x.norm_squared().exp() + 3.0,
    );
    let init = Vector2::new(0.0, 0.0);
    let eps = 1e-4;
    let search_lam = |f_nextx: &dyn Fn(f64) -> f64| om_gs::search(0.0..1.0, eps, f_nextx);
    let x = om_sd::search_2d(init, eps, f, grad, search_lam);
    println!("Steepest descent search");
    println!("x : {{{}, {}}}", x[0], x[1]);
    println!("J1: {}", f(x));
    println!("");

    let f    = |x: Vector3<f64>| x[0].powi(4) + x[1].powi(4) + (x[0] * x[1]).powi(2) + (5.0 + x[1].powi(2) + 2.0 * x[2].powi(2)).sqrt() + x[0] + x[2];
    let grad = |x: Vector3<f64>| Vector3::new(
        4.0 * x[0].powi(3) + 2.0 * x[0] * x[1].powi(2) + 1.0,
        4.0 * x[1].powi(3) + 2.0 * x[0].powi(2) * x[1] + x[1] / (5.0 + x[1].powi(2) + 2.0 * x[2].powi(2)).sqrt(),
        2.0 * x[2] / (5.0 + x[1].powi(2) + 2.0 * x[2].powi(2)).sqrt() + 1.0,
    );
    let init = Vector3::new(0.0, 0.0, 0.0);
    let eps = 1e-4;
    let search_lam = |f_nextx: &dyn Fn(f64) -> f64| om_gs::search(0.0..1.0, eps, f_nextx);
    let x = om_sd::search_3d(init, eps, f, grad, search_lam);
    println!("Steepest descent search");
    println!("x : {{{}, {}, {}}}", x[0], x[1], x[2]);
    println!("J2: {}", f(x));
    println!("");
}

fn hooke_jeeves_with_both() {
    let f = |x: Vector2<f64>| x[0].powi(2) + x.norm_squared().exp() + 4.0 * x[0] + 3.0 * x[1];
    let init_x = Vector2::new(0.0, 0.0);
    let init_per = 1.0;
    let eps = 1e-4;
    let x = om_hj::search_2d(init_x, init_per, eps, f);
    println!("Hooke-Jeeves search");
    println!("x : {{{}, {}}}", x[0], x[1]);
    println!("J1: {}", f(x));
    println!("");

    let f = |x: Vector3<f64>| x[0].powi(4) + x[1].powi(4) + (x[0] * x[1]).powi(2) + (5.0 + x[1].powi(2) + 2.0 * x[2].powi(2)).sqrt() + x[0] + x[2];
    let init_x = Vector3::new(0.0, 0.0, 0.0);
    let init_per = 1.0;
    let eps = 1e-4;
    let x = om_hj::search_3d(init_x, init_per, eps, f);
    println!("Hooke-Jeeves search");
    println!("x : {{{}, {}, {}}}", x[0], x[1], x[2]);
    println!("J2: {}", f(x));
    println!("");
}

fn main() {
    steepest_descent_with_both();
    hooke_jeeves_with_both();

    let f    = |x: Vector2<f64>| x[0].powi(2) + x.norm_squared().exp() + 4.0 * x[0] + 3.0 * x[1];
    let grad = |x: Vector2<f64>| Vector2::new(
        2.0 * x[0] * (1.0 + x.norm_squared().exp()) + 4.0,
        2.0 * x[1] * x.norm_squared().exp() + 3.0,
    );
    let init = Vector2::new(0.0, 0.0);
    let init_per = 1.0;
    let eps = 1e-4;
    let search_lam = |f_nextx: &dyn Fn(f64) -> f64| om_gs::search(0.0..1.0, eps, f_nextx);
    let etalon_x_norm = Vector2::new(-0.6132254270554607f64, -0.6632931929985874).norm();
    let mut sd_file = File::create("sd.txt").unwrap();
    let mut hj_file = File::create("hj.txt").unwrap();
    for n in 1..=300 {
        let x = om_sd::search_with_n_2d(init, n, f, grad, search_lam);
        writeln!(&mut sd_file, "{}\t{}", n, (x.norm() - etalon_x_norm).abs()).unwrap();
        let x = om_hj::search_with_n_2d(init, init_per, n, f);
        writeln!(&mut hj_file, "{}", (x.norm() - etalon_x_norm).abs()).unwrap();
    }
}
