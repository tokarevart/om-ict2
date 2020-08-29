use om_sd::na;
use na::{Vector2, Vector3};

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
    println!("x : {{{}, {}, {}}}", x[0], x[1], x[2]);
    println!("J2: {}", f(x));
    println!("");
}

fn main() {
    steepest_descent_with_both();
}
