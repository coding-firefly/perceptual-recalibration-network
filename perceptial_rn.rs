use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy)]
struct HeartData {
    age: f32,
    sex: f32,
    cp: f32,
    trestbps: f32,
    chol: f32,
    fbs: f32,
    restecg: f32,
    thalach: f32,
    exang: f32,
    oldpeak: f32,
    slope: f32,
    ca: f32,
    thal: f32,
    target: f32,
}

impl HeartData {
    fn get_value(&self, col_name: &str) -> f32 {
        match col_name {
            "age" => self.age,
            "sex" => self.sex,
            "cp" => self.cp,
            "trestbps" => self.trestbps,
            "chol" => self.chol,
            "fbs" => self.fbs,
            "restecg" => self.restecg,
            "thalach" => self.thalach,
            "exang" => self.exang,
            "oldpeak" => self.oldpeak,
            "slope" => self.slope,
            "ca" => self.ca,
            "thal" => self.thal,
            "target" => self.target,
            _ => 0.0,
        }
    }

    fn to_csv_string(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.age, self.sex, self.cp, self.trestbps, self.chol, self.fbs, 
            self.restecg, self.thalach, self.exang, self.oldpeak, self.slope, 
            self.ca, self.thal, self.target
        )
    }
}

#[derive(Debug, Clone)]
enum Node {
    Leaf(bool),
    Branch {
        column: String,
        threshold: f32,
        left: Box<Node>,
        right: Box<Node>,
    },
}

#[derive(Debug, Clone)]
struct DecisionTree {
    uuid: String,
    path: Vec<String>,
    cf_rate: f32,
    root: Node,
}

#[derive(Debug, Clone)]
struct PatientRecord {
    case_number: String,
    name: String,
    data: HeartData,
    tree_uuids: Vec<String>,
    result: bool,
    verify: Option<bool>,
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new() -> Self {
        let start = SystemTime::now();
        let since_the_epoch = start.duration_since(UNIX_EPOCH).unwrap();
        let seed = since_the_epoch.as_nanos() as u64;
        SimpleRng { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = self.state;
        (x as f64) / (u64::MAX as f64)
    }

    fn generate_uuid(&mut self) -> String {
        let v1 = self.state.wrapping_mul(3);
        let v2 = self.state.wrapping_mul(7);
        format!("{:x}-{:x}", v1, v2)
    }
}

fn shuffle_data(data: &mut Vec<HeartData>, rng: &mut SimpleRng) {
    let n = data.len();
    for i in 0..n {
        let r = (rng.next_f64() * (n as f64)) as usize;
        let r = if r >= n { n - 1 } else { r };
        data.swap(i, r);
    }
}

fn parse_line(line: &str) -> Option<HeartData> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() < 14 { return None; }
    
    let p = |idx: usize| parts[idx].trim().parse::<f32>().unwrap_or(0.0);

    Some(HeartData {
        age: p(0), sex: p(1), cp: p(2), trestbps: p(3), chol: p(4), fbs: p(5),
        restecg: p(6), thalach: p(7), exang: p(8), oldpeak: p(9), slope: p(10),
        ca: p(11), thal: p(12), target: p(13),
    })
}

fn read_and_split_csv(file_path: &str) -> (Vec<HeartData>, Vec<HeartData>) {
    let file = File::open(file_path).expect("Could not open heart.csv");
    let reader = BufReader::new(file);
    let mut all_data = Vec::new();
    
    for (i, line) in reader.lines().enumerate() {
        if i == 0 { continue; }
        if let Ok(l) = line {
            if let Some(record) = parse_line(&l) {
                all_data.push(record);
            }
        }
    }

    let total_count = all_data.len();
    let split_idx = (total_count as f32 * 0.75) as usize;
    let mut ori = Vec::new();
    let mut calibration = Vec::new();

    for (i, item) in all_data.into_iter().enumerate() {
        if i < split_idx { ori.push(item); } else { calibration.push(item); }
    }

    println!("Data Loaded: Total {}, Ori {}, Calibration {}", total_count, ori.len(), calibration.len());
    (ori, calibration)
}

fn process_ori_to_mix(mut ori: Vec<HeartData>) -> Vec<HeartData> {
    let mut rng = SimpleRng::new();
    shuffle_data(&mut ori, &mut rng);
    ori
}

fn calculate_mean(data: &[HeartData], col_name: &str) -> f32 {
    let sum: f32 = data.iter().map(|d| d.get_value(col_name)).sum();
    if data.is_empty() { 0.0 } else { sum / data.len() as f32 }
}

fn calculate_gini(data: &[HeartData]) -> f32 {
    if data.is_empty() { return 0.0; }
    let count_ones = data.iter().filter(|d| d.target == 1.0).count() as f32;
    let count_zeros = data.len() as f32 - count_ones;
    let total = data.len() as f32;
    let p_one = count_ones / total;
    let p_zero = count_zeros / total;
    1.0 - (p_one * p_one + p_zero * p_zero)
}

fn build_tree_recursive(data: Vec<HeartData>, depth: usize, path_recorder: &mut Vec<String>) -> Node {
    let count_ones = data.iter().filter(|d| d.target == 1.0).count();
    if count_ones == data.len() { return Node::Leaf(true); }
    if count_ones == 0 { return Node::Leaf(false); }
    
    if data.len() < 2 || depth > 5 {
        return Node::Leaf(count_ones > data.len() / 2);
    }

    let columns = vec!["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"];
    let mut best_gini = 1.0;
    let mut best_col = "";
    let mut best_mean = 0.0;
    let mut best_sets = (Vec::new(), Vec::new());

    for col in &columns {
        let mean_val = calculate_mean(&data, col);
        let (left_set, right_set): (Vec<HeartData>, Vec<HeartData>) = data.iter().copied().partition(|d| d.get_value(col) < mean_val);
        
        if left_set.is_empty() || right_set.is_empty() { continue; }

        let w_l = left_set.len() as f32 / data.len() as f32;
        let w_r = right_set.len() as f32 / data.len() as f32;
        let current_gini = w_l * calculate_gini(&left_set) + w_r * calculate_gini(&right_set);

        if current_gini < best_gini {
            best_gini = current_gini;
            best_col = col;
            best_mean = mean_val;
            best_sets = (left_set, right_set);
        }
    }

    if best_col == "" {
        return Node::Leaf(count_ones > data.len() / 2);
    }

    if !path_recorder.contains(&best_col.to_string()) {
        path_recorder.push(best_col.to_string());
    }

    Node::Branch {
        column: best_col.to_string(),
        threshold: best_mean,
        left: Box::new(build_tree_recursive(best_sets.0, depth + 1, path_recorder)),
        right: Box::new(build_tree_recursive(best_sets.1, depth + 1, path_recorder)),
    }
}

fn generate_forest_and_save(mix: Vec<HeartData>, calibration: &Vec<HeartData>, filename: &str, existing_trees: Option<Vec<DecisionTree>>) {
    let mut trees = if let Some(t) = existing_trees { t } else { Vec::new() };
    let mut rng = SimpleRng::new();
    let chunk_size = 25;
    let chunks: Vec<&[HeartData]> = mix.chunks(chunk_size).collect();

    println!("Building trees...");
    for chunk in chunks {
        if chunk.len() < chunk_size { continue; }
        let mut path_cols = Vec::new();
        let root_node = build_tree_recursive(chunk.to_vec(), 0, &mut path_cols);
        trees.push(DecisionTree {
            uuid: rng.generate_uuid(),
            path: path_cols,
            cf_rate: 0.5, 
            root: root_node,
        });
    }

    if !calibration.is_empty() {
        println!("Calibrating {} trees against {} proof-check records...", trees.len(), calibration.len());
        
        for tree in &mut trees {
            let mut correct_count = 0;
            for record in calibration {
                let (prediction, _) = traverse_tree(&tree.root, record);
                let actual = record.target == 1.0;
                if prediction == actual {
                    correct_count += 1;
                }
            }
            let accuracy = correct_count as f32 / calibration.len() as f32;
            tree.cf_rate = accuracy;
        }
        println!("Calibration complete. Tree weights adjusted based on accuracy.");
    }

    save_forest_to_file(&trees, filename);
    println!("Generated/Updated {} trees and saved to {}.", trees.len(), filename);
}

fn save_forest_to_file(trees: &[DecisionTree], filename: &str) {
    let mut file = File::create(filename).expect("Unable to create forest file");
    for tree in trees {
        let path_str = tree.path.join(",");
        let node_str = serialize_node(&tree.root);
        writeln!(file, "{}|{}|{}|{}", tree.uuid, tree.cf_rate, path_str, node_str).unwrap();
    }
}

fn serialize_node(node: &Node) -> String {
    match node {
        Node::Leaf(val) => format!("L({})", val),
        Node::Branch { column, threshold, left, right } => {
            format!("B({}:{}?{}:{})", column, threshold, serialize_node(left), serialize_node(right))
        }
    }
}

fn load_forest_from_file(filename: &str) -> Vec<DecisionTree> {
    if !Path::new(filename).exists() { return Vec::new(); }
    let file = File::open(filename).expect("Unable to open forest file");
    let reader = BufReader::new(file);
    let mut trees = Vec::new();

    for line in reader.lines() {
        if let Ok(l) = line {
            let parts: Vec<&str> = l.split('|').collect();
            if parts.len() < 4 { continue; }
            
            let uuid = parts[0].to_string();
            let cf_rate = parts[1].parse::<f32>().unwrap_or(0.5);
            let path: Vec<String> = parts[2].split(',').map(|s| s.to_string()).collect();
            let root = deserialize_node(parts[3]);
            
            trees.push(DecisionTree { uuid, path, cf_rate, root });
        }
    }
    trees
}

fn deserialize_node(data: &str) -> Node {
    if data.starts_with("L(") {
        let inner = &data[2..data.len()-1];
        return Node::Leaf(inner == "true");
    } else if data.starts_with("B(") {
        let inner = &data[2..data.len()-1];
        let q_index = find_split_char(inner, '?');
        let condition_part = &inner[0..q_index];
        let rest = &inner[q_index+1..];
        
        let c_parts: Vec<&str> = condition_part.split(':').collect();
        let col = c_parts[0].to_string();
        let val = c_parts[1].parse::<f32>().unwrap();
        
        let split_idx = find_split_char(rest, ':');
        let left_str = &rest[0..split_idx];
        let right_str = &rest[split_idx+1..];
        
        return Node::Branch {
            column: col,
            threshold: val,
            left: Box::new(deserialize_node(left_str)),
            right: Box::new(deserialize_node(right_str)),
        };
    }
    Node::Leaf(false)
}

fn find_split_char(s: &str, target: char) -> usize {
    let mut depth = 0;
    for (i, c) in s.chars().enumerate() {
        if c == '(' { depth += 1; }
        else if c == ')' { depth -= 1; }
        else if c == target && depth == 0 { return i; }
    }
    0
}

fn random_forest_predict(input: &HeartData, trees: &[DecisionTree]) -> (bool, Vec<(String, usize)>, Vec<String>) {
    let mut weighted_vote_true = 0.0;
    let mut weighted_vote_false = 0.0;
    let mut col_occurrences: HashMap<String, usize> = HashMap::new();
    let mut contributing_trees = Vec::new();

    for tree in trees {
        let (res, path_taken) = traverse_tree(&tree.root, input);
        if res { weighted_vote_true += tree.cf_rate; } 
        else { weighted_vote_false += tree.cf_rate; }
        
        for col in path_taken {
            *col_occurrences.entry(col).or_insert(0) += 1;
        }
    }

    let final_result = weighted_vote_true > weighted_vote_false;
    for tree in trees {
        let (res, _) = traverse_tree(&tree.root, input);
        if res == final_result { contributing_trees.push(tree.uuid.clone()); }
    }

    let mut list: Vec<(String, usize)> = col_occurrences.into_iter().collect();
    list.sort_by(|a, b| b.1.cmp(&a.1));
    (final_result, list, contributing_trees)
}

fn traverse_tree(node: &Node, input: &HeartData) -> (bool, Vec<String>) {
    match node {
        Node::Leaf(res) => (*res, Vec::new()),
        Node::Branch { column, threshold, left, right } => {
            let val = input.get_value(column);
            let mut path;
            let res;
            
            if val < *threshold {
                let r = traverse_tree(left, input);
                res = r.0;
                path = r.1;
            } else {
                let r = traverse_tree(right, input);
                res = r.0;
                path = r.1;
            }
            path.insert(0, column.clone());
            (res, path)
        }
    }
}

fn append_patient_record(record: &PatientRecord) {
    let mut file = OpenOptions::new().write(true).append(true).create(true).open("recorder.gsr").expect("Cannot open recorder.gsr");
    let tree_str = record.tree_uuids.join(";");
    let verify_str = match record.verify {
        Some(true) => "true",
        Some(false) => "false",
        None => "none",
    };
    writeln!(file, "{}|{}|{}|{}|{}|{}", record.case_number, record.name, record.data.to_csv_string(), tree_str, record.result, verify_str).unwrap();
}

fn load_patient_records() -> Vec<PatientRecord> {
    if !Path::new("recorder.gsr").exists() { return Vec::new(); }
    let file = File::open("recorder.gsr").expect("Cannot open recorder");
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for line in reader.lines() {
        if let Ok(l) = line {
            let parts: Vec<&str> = l.split('|').collect();
            if parts.len() < 6 { continue; }
            let case = parts[0].to_string();
            let name = parts[1].to_string();
            let data = parse_line(parts[2]).unwrap();
            let trees = parts[3].split(';').map(|s| s.to_string()).collect();
            let res = parts[4] == "true";
            let ver = match parts[5] {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            };
            records.push(PatientRecord { case_number: case, name, data, tree_uuids: trees, result: res, verify: ver });
        }
    }
    records
}

fn save_all_records(records: &[PatientRecord]) {
    let mut file = File::create("recorder.gsr").expect("Cannot recreate recorder");
    for rec in records {
        let tree_str = rec.tree_uuids.join(";");
        let verify_str = match rec.verify {
            Some(true) => "true",
            Some(false) => "false",
            None => "none",
        };
        writeln!(file, "{}|{}|{}|{}|{}|{}", rec.case_number, rec.name, rec.data.to_csv_string(), tree_str, rec.result, verify_str).unwrap();
    }
}

fn feedback_loop() {
    println!("\n--- FEEDBACK MODE ---");
    let mut records = load_patient_records();
    let mut trees = load_forest_from_file("forest.gsh");

    if records.is_empty() { println!("No records found."); return; }

    println!("Enter Case Number to verify:");
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let case_query = input.trim();

    let mut target_idx = None;
    for (i, rec) in records.iter().enumerate() {
        if rec.case_number == case_query {
            target_idx = Some(i);
            break;
        }
    }

    if let Some(idx) = target_idx {
        let rec = &records[idx];
        println!("Case: {}, Result: {}, Verify: {:?}", rec.case_number, rec.result, rec.verify);
        println!("Change Verify Status? (t=true/f=false/n=no change):");
        
        let mut choice = String::new();
        io::stdin().read_line(&mut choice).unwrap();
        let new_status = match choice.trim() {
            "t" => Some(true),
            "f" => Some(false),
            _ => return,
        };

        records[idx].verify = new_status;

        if new_status == Some(true) {
            println!("Verification Positive. Updating Tree Confidence Rates...");
            let winning_uuids = &records[idx].tree_uuids;
            let win_count = winning_uuids.len() as f32;
            let increment = if win_count > 0.0 { 0.1 / win_count } else { 0.0 };
            
            for tree in &mut trees {
                if winning_uuids.contains(&tree.uuid) {
                    tree.cf_rate = (tree.cf_rate + increment).min(1.0);
                } else {
                    tree.cf_rate = (tree.cf_rate - 0.01).max(0.0);
                }
            }
        }

        let verified_indices: Vec<usize> = records.iter().enumerate()
            .filter(|(_, r)| r.verify == Some(true))
            .map(|(i, _)| i)
            .collect();

        if verified_indices.len() >= 25 {
            println!("25 Verified records found. Generating new tree and pruning forest...");
            let mut new_data = Vec::new();
            for &v_idx in &verified_indices[0..25] {
                new_data.push(records[v_idx].data);
            }
            
            let mut rng = SimpleRng::new();
            let mut path_cols = Vec::new();
            let new_root = build_tree_recursive(new_data, 0, &mut path_cols);
            let new_tree = DecisionTree {
                uuid: rng.generate_uuid(),
                path: path_cols,
                cf_rate: 0.5,
                root: new_root
            };
            trees.push(new_tree);

            trees.sort_by(|a, b| a.cf_rate.partial_cmp(&b.cf_rate).unwrap());
            if !trees.is_empty() {
                println!("Pruning weakest tree UUID: {} (CF: {})", trees[0].uuid, trees[0].cf_rate);
                trees.remove(0);
            }

            save_forest_to_file(&trees, "forest.gsh");
            let used_cases: Vec<String> = verified_indices[0..25].iter().map(|&i| records[i].case_number.clone()).collect();
            records.retain(|r| !used_cases.contains(&r.case_number));
            println!("Cycle complete. Forest updated. Records cleaned.");
        } else {
            save_forest_to_file(&trees, "forest.gsh");
        }
        
        save_all_records(&records);

    } else {
        println!("Case not found.");
    }
}

fn main() {
    if !Path::new("heart.csv").exists() {
        println!("heart.csv not found. Creating dummy file...");
        let dummy_data = "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target\n63,1,3,145,233,1,0,150,0,2.3,0,0,1,1\n37,1,2,130,250,0,1,187,0,3.5,0,0,2,1\n41,0,1,130,204,0,0,172,0,1.4,2,0,2,1\n56,1,1,120,236,0,0,178,0,0.8,2,0,2,1\n57,0,0,120,354,0,1,163,1,0.6,2,0,2,1\n57,1,0,140,192,0,1,148,0,0.4,1,0,1,1\n56,0,1,140,294,0,0,153,0,1.3,1,0,2,1\n44,1,1,120,263,0,1,173,0,0.0,2,0,3,1\n52,1,2,172,199,1,1,162,0,0.5,2,0,3,1\n57,1,2,150,168,0,1,174,0,1.6,2,0,2,1\n54,1,0,140,239,0,1,160,0,1.2,2,0,2,1\n48,0,2,130,275,0,1,139,0,0.2,2,0,2,1\n49,1,1,130,266,0,1,171,0,0.6,2,0,2,1\n64,1,3,110,211,0,0,144,1,1.8,1,0,2,1\n58,0,3,150,283,1,0,162,0,1.0,2,0,2,1\n50,0,2,120,219,0,1,158,0,1.6,1,0,2,1\n58,0,2,120,340,0,1,172,0,0.0,2,0,2,1\n66,0,3,150,226,0,1,114,0,2.6,0,0,2,1\n43,1,0,150,247,0,1,171,0,1.5,2,0,2,1\n69,0,3,140,239,0,1,151,0,1.8,2,2,2,1\n59,1,0,135,234,0,1,161,0,0.5,1,0,3,1\n44,1,2,130,233,0,1,179,1,0.4,2,0,2,1\n42,1,0,140,226,0,1,178,0,0.0,2,0,2,1\n61,1,2,150,243,1,1,137,1,1.0,1,0,2,1\n40,1,3,140,199,0,1,178,1,1.4,2,0,3,1\n71,0,1,160,302,0,1,162,0,0.4,2,2,2,1";
        let mut f = File::create("heart.csv").expect("Unable to create file");
        f.write_all(dummy_data.as_bytes()).expect("Unable to write data");
    }else{
        println!("heart.csv detected, initialising..")
    }

    if !Path::new("forest.gsh").exists() {
        println!("Training Decision Tree from Data");
        let (ori, calibration) = read_and_split_csv("heart.csv");
        let mix = process_ori_to_mix(ori);
        generate_forest_and_save(mix, &calibration, "forest.gsh", None);
    }

    println!("\n=== MEDICAL AI TERMINAL ===");
    println!("1. Enter New Patient Data & Predict");
    println!("2. Verify/Feedback Mode");
    println!("3. Exit");
    print!("Select Option: ");
    io::stdout().flush().unwrap();
    
    let mut mode = String::new();
    io::stdin().read_line(&mut mode).unwrap();
    
    match mode.trim() {
        "1" => {
            println!("\n--- NEW PATIENT ENTRY ---");

            print!("Enter Case Number: ");
            io::stdout().flush().unwrap();
            let mut case_num = String::new();
            io::stdin().read_line(&mut case_num).unwrap();
            let case_num = case_num.trim().to_string();

            print!("Enter Patient Name: ");
            io::stdout().flush().unwrap();
            let mut p_name = String::new();
            io::stdin().read_line(&mut p_name).unwrap();
            let p_name = p_name.trim().to_string();

            let read_float = |prompt: &str| -> f32 {
                print!("{}: ", prompt);
                io::stdout().flush().unwrap();
                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                input.trim().parse::<f32>().unwrap_or(0.0)
            };

            let age = read_float("Age");
            let sex = read_float("Sex (1=Male, 0=Female)");
            let cp = read_float("Chest Pain Type (0-3)");
            let trestbps = read_float("Resting Blood Pressure");
            let chol = read_float("Cholesterol");
            let fbs = read_float("Fasting Blood Sugar > 120 (1=True, 0=False)");
            let restecg = read_float("Resting ECG (0-2)");
            let thalach = read_float("Max Heart Rate");
            let exang = read_float("Exercise Angina (1=Yes, 0=No)");
            let oldpeak = read_float("Oldpeak (ST depression)");
            let slope = read_float("Slope (0-2)");
            let ca = read_float("Major Vessels (0-3)");
            let thal = read_float("Thal (1=Fixed, 2=Normal, 3=Reversable)");
            let target = 0.0; 

            let input_data = HeartData {
                age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
            };

            println!("\nProcessing Forest...");
            let trees = load_forest_from_file("forest.gsh");
            let (result, path_counts, contributing_trees) = random_forest_predict(&input_data, &trees);
            
            println!("-----------------------------");
            println!("PREDICTION: {}", if result { "POSITIVE (1)" } else { "NEGATIVE (0)" });
            println!("-----------------------------");
            
            println!("Most Influential Factors (Top 3):");
            for (i, (col, count)) in path_counts.iter().enumerate() {
                if i >= 3 { break; }
                println!(" - {}: {} checks", col, count);
            }

            let record = PatientRecord {
                case_number: case_num,
                name: p_name,
                data: input_data,
                tree_uuids: contributing_trees,
                result,
                verify: None 
            };
            
            append_patient_record(&record);
            println!("\nRecord saved. Pending verification in Mode 2.");
        },
        "2" => {
            feedback_loop();
        },
        _ => println!("Exiting."),
    }
}
