/*
 *  DelphiOffline's C interface 
 *
 *  Created on: June 10, 2019
 *      Author: ryanleh
 */
#include <iomanip>
#include <math.h>
#include "interface.h"
#include "conv2d.h"
#include "fc_layer.h"
#include "triples_gen.h"

/* Formatted printed for batched plaintext */
inline void print_batch(size_t slot_count, size_t print_size, uv64 &plain) {
    size_t row_size = slot_count / 2;
    cout << endl;

    cout << "    [";
    for (size_t i = 0; i < print_size; i++)
    {
        cout << setw(3) << plain[i] << ",";
    }
    cout << setw(3) << " ...,";
    for (size_t i = row_size - print_size; i < row_size; i++)
    {
        cout << setw(3) << plain[i] << ((i != row_size - 1) ? "," : " ]\n");
    }
    cout << "    [";
    for (size_t i = row_size; i < row_size + print_size; i++)
    {
        cout << setw(3) << plain[i] << ",";
    }
    cout << setw(3) << " ...,";
    for (size_t i = 2 * row_size - print_size; i < 2 * row_size; i++)
    {
        cout << setw(3) << plain[i] << ((i != 2 * row_size - 1) ? "," : " ]\n");
    }
    cout << endl;
}

/* Serializes an outputstream into a byte array.
 * Returns the bytearray and size */
SerialCT serialize(ostringstream &os) {
    char* serialized = new char[os.tellp()];
    string tmp_str = os.str();
    std::move(tmp_str.begin(), tmp_str.end(), serialized);
    return SerialCT { serialized, (uint64_t) os.tellp() };
}

/* Serializes a SEAL ciphertext to a byte array. 
 * returns the bytearray and size */
SerialCT serialize_ct(vector<Ciphertext> ct_vec) {
    ostringstream os;
    for (auto &ct: ct_vec)
        ct.save(os);
    return serialize(os);
}

/* Extracts a vector of Ciphertexts from provided byte stream */
void recast_opaque(SerialCT &ct, vector<Ciphertext>& destination,
        SEALContext* context) {
    istringstream is;
    is.rdbuf()->pubsetbuf(ct.inner, ct.size);
    for(int ct_idx = 0; ct_idx < destination.size(); ct_idx++) {
        destination[ct_idx].load(*context, is);
    }
}

/* Encodes a vector of u64 into SEAL Plaintext */
vector<Plaintext> encode_vec(const u64* shares, u64 num, BatchEncoder& encoder) {
    u64 slot_count = encoder.slot_count();
    int vec_size = ceil((float)num / slot_count);
    vector<Plaintext> result(vec_size);

#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int vec_idx = 0; vec_idx < vec_size; vec_idx++) {
        uv64 pod_matrix(slot_count, 0ULL);
        int limit = min(num-vec_idx*slot_count, slot_count);
        for (int i = 0; i < limit; i++) {
            pod_matrix[i] = shares[vec_idx*slot_count + i];
        }
        encoder.encode(pod_matrix, result[vec_idx]);
    }
    return result;
}

/* Encrypts and serializes a vector */
SerialCT encrypt_vec(const ClientFHE* cfhe, const u64* vec, u64 vec_size) {
    // Recast the needed fhe helpers
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
   
    // Encrypt vec
    auto pt_vec = encode_vec(vec, vec_size, *encoder);
    vector<Ciphertext> ct_vec(pt_vec.size());
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < pt_vec.size(); i++) {
        encryptor->encrypt(pt_vec[i], ct_vec[i]);
    }

    // Serialize ciphertexts
    SerialCT ct = serialize_ct(ct_vec);
    return ct;
}

/* Deserializes and decrypts a vector */
u64* decrypt_vec(const ClientFHE* cfhe, SerialCT *ct, u64 size) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(cfhe->context);

    // Recast needed fhe helpers
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);

    // Recast bytearrays to vectors of Ciphertexts and decrypt
    u64 slot_count = encoder->slot_count();
    u64 vec_size = ceil((float)size / slot_count);
    vector<Ciphertext> ct_vec(vec_size);
    recast_opaque(*ct, ct_vec, context);
    
    // Decrypt ciphertext 
    u64* share = new u64[size];
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < vec_size; i++) {
        vector<u64> pod_matrix(slot_count, 0ULL);
        Plaintext tmp;
        decryptor->decrypt(ct_vec[i], tmp);
        encoder->decode(tmp, pod_matrix);
        for (int j = 0; j < min(slot_count, size - slot_count*i); j++) {
            share[slot_count*i + j] = pod_matrix[j];
        }
    }
    return share;
}

ClientFHE client_keygen(SerialCT *key_share) {
    //---------------Param and Key Generation---------------
    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(POLY_MOD_DEGREE);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(POLY_MOD_DEGREE));
    parms.set_plain_modulus(PLAINTEXT_MODULUS);
    auto context = new SEALContext(parms);
    KeyGenerator keygen(*context);
    auto sec_key = keygen.secret_key();
    // Get serialized versions of the keys
    auto ser_pub_key = keygen.create_public_key();
    auto ser_gal_keys = keygen.create_galois_keys();
    auto ser_relin_keys = keygen.create_relin_keys();
    // Deserialize the public key since we use it when creating the local
    // objects
    PublicKey pub_key;
    stringstream pub_key_s;
    ser_pub_key.save(pub_key_s);
    pub_key.load(*context, pub_key_s);

    BatchEncoder *encoder = new BatchEncoder(*context);
    Encryptor *encryptor = new Encryptor(*context, pub_key);
    Evaluator *evaluator = new Evaluator(*context);
    Decryptor *decryptor = new Decryptor(*context, sec_key); 

    // Recast the context to void*
    void* void_context = static_cast<void*>(context);

    // Serialize params and all keys
    ostringstream os;
    parms.save(os);
    ser_pub_key.save(os);
    ser_gal_keys.save(os);
    ser_relin_keys.save(os);
    *key_share = serialize(os);
    return ClientFHE { void_context, encoder, encryptor, evaluator, decryptor };
}


ServerFHE server_keygen(SerialCT key_share) {
    istringstream is;
    is.rdbuf()->pubsetbuf(key_share.inner, key_share.size);

    // Load params
    EncryptionParameters parms;
    parms.load(is);
    auto context = new SEALContext(parms);

    // Load keys
    PublicKey pub_key;
    GaloisKeys* gal_keys = new GaloisKeys();
    RelinKeys* relin_keys = new RelinKeys();
    pub_key.load(*context, is);
    (*gal_keys).load(*context, is);
    (*relin_keys).load(*context, is);

    // Create helpers
    BatchEncoder *encoder = new BatchEncoder(*context);
    Encryptor *encryptor = new Encryptor(*context, pub_key);
    Evaluator *evaluator = new Evaluator(*context);
    
    // Recast the context to void*
    void* void_context = static_cast<void*>(context);
    
    // Generate the zero ciphertext
    vector<u64> pod_matrix(encoder->slot_count(), 0ULL);
    Plaintext tmp;
    Ciphertext* zero = new Ciphertext();
    encoder->encode(pod_matrix, tmp);
    encryptor->encrypt(tmp, *zero);

    return ServerFHE { void_context, encoder, encryptor, evaluator, gal_keys, relin_keys,
      (char*) zero };
}

Metadata conv_metadata(void* batch_encoder, int32_t image_h, int32_t image_w, int32_t filter_h, int32_t filter_w,
        int32_t inp_chans, int32_t out_chans, int32_t stride_h, int32_t stride_w, bool pad_valid) {
    int slot_count = (reinterpret_cast<BatchEncoder*>(batch_encoder))->slot_count();
    return conv_metadata(slot_count, image_h, image_w, filter_h, filter_w, inp_chans, out_chans,
        stride_h, stride_w, pad_valid);
}

Metadata fc_metadata(void* batch_encoder, int32_t vector_len, int32_t matrix_h) {
    int slot_count = (reinterpret_cast<BatchEncoder*>(batch_encoder))->slot_count();
    return fc_metadata(slot_count, vector_len, matrix_h);
}

ClientShares client_conv_preprocess(const ClientFHE* cfhe, const Metadata* data, const u64* const* image) {
    // Recast the needed fhe helpers
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    
    // Preprocess image
    auto pt = preprocess_image(*data, image);
    auto rotated_pt = filter_rotations(pt, *data);
    auto ct_rotations = HE_encrypt_rotations(rotated_pt, *data, *encryptor, *encoder);

    // Flatten rotations ciphertext
    vector<Ciphertext> ct_flat_rotations;
    for (const auto &ct: ct_rotations)
        ct_flat_rotations.insert(ct_flat_rotations.end(), ct.begin(), ct.end());

    // Serialize vector
    ClientShares shares;
    shares.input_ct = serialize_ct(ct_flat_rotations);
    return shares;
}

char**** server_conv_preprocess(const ServerFHE* sfhe, const Metadata* data,
        const u64* const* const* filters) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);

    // Preprocess filters
    auto masks_vec = HE_preprocess_filters(filters, *data, *encoder);
   
    // Recast masks to use opaque pointers for C interface
    char**** masks = new char***[masks_vec.size()];
    for (int i = 0; i < masks_vec.size(); i++) {
        masks[i] = new char**[masks_vec[0].size()];
        for (int j = 0; j < masks_vec[0].size(); j++) {
            masks[i][j] = new char*[masks_vec[0][0].size()];
            for (int z = 0; z < masks_vec[0][0].size(); z++)
                masks[i][j][z] = (char*) new Plaintext(masks_vec[i][j][z]);
        }
    }
    return masks;
}

ServerShares server_conv_preprocess_shares(const ServerFHE* sfhe, const Metadata* data,
        const u64* const* linear_share) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);

    // Reshape shares
    vector<Plaintext> linear = HE_preprocess_noise(linear_share, *data, *encoder);
    
    // Recast everything back to opaque C types
    auto enc_linear_share = new char*[data->out_ct];
    for (int ct_idx = 0; ct_idx < data->out_ct; ct_idx++) {
        enc_linear_share[ct_idx] = (char*) new Plaintext(linear[ct_idx]);
    }
   
    ServerShares shares;
    shares.linear = enc_linear_share;
    return shares;
}


ClientShares client_fc_preprocess(const ClientFHE* cfhe, const Metadata* data, const u64* vector) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);

    // Preprocess input vector
    Plaintext enc_vec = preprocess_vec(*data, *encoder, vector);
    std::vector<Ciphertext> ct(1);
    encryptor->encrypt(enc_vec, ct[0]);

    // Convert vector to char array and flatten to a single byte array
    ClientShares shares;
    shares.input_ct = serialize_ct(ct);
    return shares;
}


char** server_fc_preprocess(const ServerFHE* sfhe, const Metadata* data, const u64* const* matrix) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);

    // Preprocess matrix
    vector<Plaintext> enc_matrix = preprocess_matrix(matrix, *data, *encoder);
   
    // Convert to opaque C types
    char** enc_matrix_c = new char*[data->inp_ct];
    for (int i = 0; i < data->inp_ct; i++) {
        enc_matrix_c[i] = (char*) new Plaintext(enc_matrix[i]);
    }
    return enc_matrix_c;
}


ServerShares server_fc_preprocess_shares(const ServerFHE* sfhe, const Metadata* data,
        const u64* linear_share) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);

    // Reshape shares
    Plaintext linear = fc_preprocess_noise(*data, *encoder, linear_share);
    
    // Recast shares to opaque pointers
    auto enc_linear_share = new char*[1];
    enc_linear_share[0] = (char*) new Plaintext(linear);

    ServerShares shares;
    shares.linear = enc_linear_share;
    return shares;
}

ServerTriples server_triples_preprocess(const ServerFHE* sfhe, uint32_t num_triples, const u64* a_share,
    const u64* b_share, const u64* c_share) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);

    // Encode shares
    vector<Plaintext> enc_a = encode_vec(a_share, num_triples, *encoder);
    vector<Plaintext> enc_b = encode_vec(b_share, num_triples, *encoder);
    vector<Plaintext> enc_c = encode_vec(c_share, num_triples, *encoder);
    
    // Recast shares to opaque pointers
    u64 vec_size = enc_a.size();
    char** a = new char*[vec_size]; 
    char** b = new char*[vec_size];
    char** c = new char*[vec_size];
    for (int i = 0; i < vec_size; i++) {
        a[i] = (char*) new Plaintext(enc_a[i]);
        b[i] = (char*) new Plaintext(enc_b[i]);
        c[i] = (char*) new Plaintext(enc_c[i]);
    }

    ServerTriples shares;
    shares.num = num_triples;
    shares.vec_len = vec_size;
    shares.a_share = a;
    shares.b_share = b;
    shares.c_share = c;
    return shares;
}


ClientTriples client_triples_preprocess(const ClientFHE* cfhe, uint32_t num_triples, const u64* a_rand,
        const u64* b_rand) {
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);

    // Encode randomizers
    vector<Plaintext> enc_a = encode_vec(a_rand, num_triples, *encoder);
    vector<Plaintext> enc_b = encode_vec(b_rand, num_triples, *encoder);

    // Encrypt randomizers
    u64 vec_size = enc_a.size();
    vector<Ciphertext> vec_a(vec_size);
    vector<Ciphertext> vec_b(vec_size);
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < vec_size; i++) {
        encryptor->encrypt(enc_a[i], vec_a[i]);
        encryptor->encrypt(enc_b[i], vec_b[i]);
    }

    // Recast to opaque pointers
    SerialCT a, b;
    a = serialize_ct(vec_a);
    b = serialize_ct(vec_b);

    ClientTriples shares;
    shares.num = num_triples;
    shares.vec_len = vec_size;
    shares.a_ct = a;
    shares.b_ct = b;
    return shares;
}


void server_conv_online(const ServerFHE* sfhe, const Metadata* data, SerialCT ciphertext,
    char**** masks, ServerShares* shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(sfhe->context);

    // Deserialize ciphertexts
    istringstream is;
    is.rdbuf()->pubsetbuf(ciphertext.inner, ciphertext.size);
    vector<vector<Ciphertext>> ct_vec(data->inp_ct, vector<Ciphertext>(data->filter_size));
    for (int i = 0; i < data->inp_ct; i++) {
        for (int j = 0; j < ct_vec[0].size(); j++) {
            ct_vec[i][j].load(*context, is);
        } 
    }

    // Recast opaque pointers to vectors
    vector<vector<vector<Plaintext>>> masks_vec(data->convs, 
            vector<vector<Plaintext>>(data->inp_ct, 
                vector<Plaintext>(data->filter_size)));
    for (int i = 0; i < masks_vec.size(); i++) {
        for (int j = 0; j < masks_vec[0].size(); j++) {
            for (int z = 0; z < masks_vec[0][0].size(); z++)
                masks_vec[i][j][z] = *(reinterpret_cast<Plaintext*>(masks[i][j][z]));
        } 
    }
    vector<Plaintext> linear_share(data->out_ct);
    for (int ct_idx = 0; ct_idx < data->out_ct; ct_idx++) {
        linear_share[ct_idx] = *(reinterpret_cast<Plaintext*>(shares->linear[ct_idx]));
    }

    // Recast needed fhe helpers and ciphertexts
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    GaloisKeys *gal_keys = reinterpret_cast<GaloisKeys*>(sfhe->gal_keys);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(sfhe->relin_keys);
    Ciphertext *zero = reinterpret_cast<Ciphertext*>(sfhe->zero);

    // Evaluation
    auto rotation_sets = HE_conv(masks_vec, ct_vec, *data, *evaluator, *relin_keys, *zero); 
    vector<Ciphertext> linear = HE_output_rotations(rotation_sets, *data, *evaluator, *gal_keys, *zero);

    // Secret share the result
    for (int ct_idx = 0; ct_idx < data->out_ct; ct_idx++) {
        evaluator->sub_plain_inplace(linear[ct_idx], linear_share[ct_idx]);
    }

    // Serialize the resulting ciphertexts into bytearrays and store in ServerShares
    shares->linear_ct = serialize_ct(linear);
}


void server_fc_online(const ServerFHE* sfhe, const Metadata* data, SerialCT ciphertext,
    char** matrix, ServerShares* shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(sfhe->context);

    // Deserialize ciphertext
    istringstream is;
    is.rdbuf()->pubsetbuf(ciphertext.inner, ciphertext.size);
    Ciphertext input;
    input.load(*context, is);

    // Recast opaque pointers
    vector<Plaintext> matrix_vec(data->inp_ct);
    for (int i = 0; i < data->inp_ct; i++)
        matrix_vec[i] = *(reinterpret_cast<Plaintext*>(matrix[i]));
    Plaintext linear_share = *(reinterpret_cast<Plaintext*>(shares->linear[0]));
  
    // Recast needed fhe helpers and ciphertexts
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    GaloisKeys *gal_keys = reinterpret_cast<GaloisKeys*>(sfhe->gal_keys);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(sfhe->relin_keys);
    Ciphertext *zero = reinterpret_cast<Ciphertext*>(sfhe->zero);

    // Evaluation
    vector<Ciphertext> linear(1, fc_online(input, matrix_vec, *data, *evaluator, *gal_keys, *relin_keys, *zero));

    // Linear share
    evaluator->sub_plain_inplace(linear[0], linear_share);

    // Serialize the resulting ciphertexts into bytearrays and store in ServerShares
    shares->linear_ct = serialize_ct(linear);
}


void server_triples_online(const ServerFHE* sfhe, SerialCT client_a, SerialCT client_b, ServerTriples* shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(sfhe->context);

    // Recast needed fhe helpers
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    RelinKeys *relin_keys = reinterpret_cast<RelinKeys*>(sfhe->relin_keys);

    // Recast client ciphertexts
    vector<Ciphertext> client_a_ct(shares->vec_len);
    vector<Ciphertext> client_b_ct(shares->vec_len);
    recast_opaque(client_a, client_a_ct, context);
    recast_opaque(client_b, client_b_ct, context);

    // Recast opaque pointers
    vector<Plaintext> a_share(shares->vec_len);
    vector<Plaintext> b_share(shares->vec_len); 
    vector<Plaintext> c_share(shares->vec_len);
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < shares->vec_len; i++) {
        a_share[i] = *(reinterpret_cast<Plaintext*>(shares->a_share[i]));
        b_share[i] = *(reinterpret_cast<Plaintext*>(shares->b_share[i]));
        c_share[i] = *(reinterpret_cast<Plaintext*>(shares->c_share[i]));
    }

    // Evaluation - share of c is now in c_ct
    vector<Ciphertext> c_ct(client_a_ct.size());
    triples_online(client_a_ct, client_b_ct, c_ct, a_share, b_share, c_share, *evaluator, *relin_keys);

    // Serialize the ciphertexts
    shares->c_ct = serialize_ct(c_ct);
}


void client_conv_decrypt(const ClientFHE *cfhe, const Metadata *data, ClientShares *shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(cfhe->context);

    // Recast needed fhe helpers
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);

    // Recast bytearrays to vectors of Ciphertexts and decrypt
    vector<Ciphertext> linear_ct(data->out_ct);
    recast_opaque(shares->linear_ct, linear_ct, context);
    shares->linear = HE_decrypt(linear_ct, *data, *decryptor, *encoder);
}

void client_fc_decrypt(const ClientFHE *cfhe, const Metadata *data, ClientShares *shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(cfhe->context);

    // Recast needed fhe helpers
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);

    // Recast bytearrays to vectors of Ciphertexts and decrypt
    vector<Ciphertext> linear_ct(1);
    recast_opaque(shares->linear_ct, linear_ct, context);
    shares->linear = new u64*[1];
    shares->linear[0] = fc_postprocess(linear_ct[0], *data, *encoder, *decryptor);
}

/* Decrypts the clients multiplication triple share */
void client_triples_decrypt(const ClientFHE *cfhe, SerialCT c, ClientTriples *shares) {
    // Grab shared pointer to SEALContext
    auto context = static_cast<SEALContext*>(cfhe->context);
    
    // Recast needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);

    // Recast received bytearrays to Ciphertexts
    vector<Ciphertext> c_ct(shares->vec_len);
    recast_opaque(c, c_ct, context);

    // Decrypt Ciphertexts
    shares->c_share = client_triples_postprocess(shares->num, c_ct, *encoder, *decryptor);
}

void client_free_keys(const ClientFHE* cfhe) {
    delete (BatchEncoder*) cfhe->encoder;
    delete (Encryptor*) cfhe->encryptor;
    delete (Evaluator*) cfhe->evaluator;
    delete (Decryptor*) cfhe->decryptor;

    // Delete SEALContext ptr
    auto tmp_ptr = static_cast<SEALContext*>(cfhe->context);
    delete tmp_ptr;
}

void server_free_keys(const ServerFHE *sfhe) {
    delete (BatchEncoder*) sfhe->encoder;
    delete (Encryptor*) sfhe->encryptor;
    delete (Evaluator*) sfhe->evaluator;
    delete (GaloisKeys*) sfhe->gal_keys;
    delete (RelinKeys*) sfhe->relin_keys;
    delete (Ciphertext*) sfhe->zero;

    // Delete SEALContext ptr
    auto tmp_ptr = static_cast<SEALContext*>(sfhe->context);
    delete tmp_ptr;
}

void free_ct(SerialCT *ct) {
    delete[] ct->inner;
}

void client_conv_free(const Metadata *data, ClientShares* shares) {
    // Received ciphertexts are allocated by Rust so only need to free input
    free_ct(&shares->input_ct);
    // Free shares
    for (int idx = 0; idx < data->out_chans; idx++) {
        delete[] shares->linear[idx];
    }
    delete[] shares->linear;
}


void server_conv_free(const Metadata* data, char**** masks, ServerShares* shares) {
    // Free masks
    for (int conv = 0; conv < data->convs; conv++) {
        for (int ct_idx = 0; ct_idx < data->inp_ct; ct_idx++) {
            for (int rot = 0; rot < data->filter_size; rot++) {
                delete (Plaintext*) masks[conv][ct_idx][rot]; 
            }
            delete[] masks[conv][ct_idx];
        }
        delete[] masks[conv];
    } 
    delete[] masks;
    // Free shares
    for (int ct = 0; ct < data->out_ct; ct++) {
        delete (Plaintext*) shares->linear[ct];
    }
    delete[] shares->linear;
    
    // Free ciphertexts
    free_ct(&shares->linear_ct);
}


void client_fc_free(ClientShares* shares) {
    // Received ciphertexts are allocated by Rust so only need to free input
    free_ct(&shares->input_ct);
    // Free shares
    delete[] shares->linear[0];
    delete[] shares->linear;
}


void server_fc_free(const Metadata* data, char** enc_matrix, ServerShares* shares) {
    // Free matrix
    for (int idx = 0; idx < data->inp_ct; idx++) {
        delete (Plaintext*) enc_matrix[idx];
    }
    delete[] enc_matrix;
    // Free shares
    delete (Plaintext*) shares->linear[0];
    delete[] shares->linear;
    // Free ciphertexts
    free_ct(&shares->linear_ct);
}


void client_triples_free(ClientTriples* shares) {
    // Free shares
    delete[] shares->c_share;
    // Free ciphertexts
    free_ct(&shares->a_ct);
    free_ct(&shares->b_ct);
}


void server_triples_free(ServerTriples* shares) {
    // Free vectors of Plaintexts
    for (int idx = 0; idx < shares->vec_len; idx++) {
        delete (Plaintext*) shares->a_share[idx];
        delete (Plaintext*) shares->b_share[idx];
        delete (Plaintext*) shares->c_share[idx];
    }
    delete[] shares->a_share;
    delete[] shares->b_share;
    delete[] shares->c_share;
    // Free ciphertexts
    free_ct(&shares->c_ct);
}

void add_matrix_multiplication()
{
    EncryptionParameters parms(scheme_type::bfv);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));

    /*
    To enable batching, we need to set the plain_modulus to be a prime number
    congruent to 1 modulo 2*poly_modulus_degree. Microsoft SEAL provides a helper
    method for finding such a prime. In this example we create a 20-bit prime
    that supports batching.
    */
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));

    SEALContext context(parms);

    /*
    We can verify that batching is indeed enabled by looking at the encryption
    parameter qualifiers created by SEALContext.
    */
    auto qualifiers = context.first_context_data()->qualifiers();
    cout << "Batching enabled: " << boolalpha << qualifiers.using_batching << endl;

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    /*
    Batching is done through an instance of the BatchEncoder class.
    */
    BatchEncoder batch_encoder(context);
    
    size_t slot_count = batch_encoder.slot_count();
    size_t row_size = slot_count / 2;

    clock_t start = clock();
    // Difine matrix X
    size_t a_dim = 1021; 
    vector<uint64_t> x_row_1(slot_count, 0ULL);
    vector<vector<uint64_t>> x_row(a_dim,x_row_1);

    /*
    Encrypt the Matrix X
    */

    vector<Plaintext> plain_x_row(a_dim);
    vector<Ciphertext> encrypted_x_row(a_dim);
    for (size_t i = 0; i < a_dim; i++){
        batch_encoder.encode(x_row[i],plain_x_row[i]);
        encryptor.encrypt(plain_x_row[i], encrypted_x_row[i]);
    }

    // Define the adjaeceny matrix
    
    int a[42][42] = {
        {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
        {0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0},
        {0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}};

    // Define a zero Ciphertext
    vector<uint64_t> pod_matrix(slot_count, 0ULL);
    Plaintext tmp;
    Ciphertext zero;
    batch_encoder.encode(pod_matrix, tmp);
    encryptor.encrypt(tmp, zero);

    // 使用add_inplace需要初始化
    // vector<Ciphertext> encrypted_result(a_dim,zero);

    // 使用add_many不需要初始化
    // 实际使用过程中不能使用add_may,因为可能存在邻接矩阵A的一行全是0的情况,这种情况下，结果矩阵的该行将没有结果
    vector<Ciphertext> encrypted_result(a_dim);

    // Compute the Matrix Multiplication
    // for (size_t i = 0; i < a_dim; i++){
    //     vector<Ciphertext> result_ct;
    //     for (size_t j = 0; j < a_dim; j++){
    //         if(a[i][j]==1){
    //             result_ct.push_back(encrypted_x_row[j]);
    //             // evaluator.add_inplace(encrypted_result[i],encrypted_x_row[j]);
    //         }
    //     }
    //     evaluator.add_many(result_ct,encrypted_result[i]);
    // }
    for (size_t i = 0; i < a_dim; i++){
        vector<Ciphertext> result_ct;
        for (size_t j = 0; j < a_dim; j++){
            if(j<3){
                cout << i << "," << j << endl;
                result_ct.push_back(encrypted_x_row[j]);
                // evaluator.add_inplace(encrypted_result[i],encrypted_x_row[j]);
            }
        }
        evaluator.add_many(result_ct,encrypted_result[i]);
    }
    
    vector<Plaintext> plain_result(a_dim);
    vector<vector<uint64_t>> pod_result(a_dim);
    
    for (size_t i = 0; i < a_dim; i++){
        decryptor.decrypt(encrypted_result[i], plain_result[i]);
        batch_encoder.decode(plain_result[i], pod_result[i]);
    }
    clock_t end = clock();
    cout << double(end - start) / CLOCKS_PER_SEC << "s " <<  endl; // 打印0.001s

}

// 后面可以给这个函数改一下名字，e.g. encrypt_matrix_row 或者 encrypt_vec
SerialCT server_aggregation(const ClientFHE* cfhe, uint64_t a_dim, uint64_t hidden_dim, const u64* const* matrix){
    // Recast the needed fhe helpers
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(cfhe->encryptor);

    // 这个8192是slot_count
    uv64 pod_matrix(8192, 0ULL);

    vector<Ciphertext> ct(a_dim);
    for(int col = 0; col < a_dim; col++){
        for(int row = 0; row < hidden_dim; row++){
            pod_matrix[row] = matrix[col][row];
            // cout << "[" << col << "," << row << "] "<< matrix[col][row] << ",##############" << endl;
        }
        // Encode matrix
        Plaintext enc_vec;
        encoder->encode(pod_matrix, enc_vec);
        encryptor->encrypt(enc_vec, ct[col]);
    }

    // Serialize ciphertexts
    SerialCT serial_ct = serialize_ct(ct);
    return serial_ct;
}

u64** server_decrypt_ax(const ClientFHE* cfhe, uint64_t a_dim, uint64_t hidden_dim, SerialCT ct){
    auto context = static_cast<SEALContext*>(cfhe->context);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(cfhe->encoder);
    Decryptor *decryptor = reinterpret_cast<Decryptor*>(cfhe->decryptor);

    // 将ct还原为a_dim个密文
    vector<Ciphertext> x_ct(a_dim);
    recast_opaque(ct, x_ct, context);

    Plaintext plain_result;
    vector<uv64> pod_result(a_dim);

    for(int i = 0; i < a_dim; i++){
        decryptor->decrypt(x_ct[i], plain_result);
        encoder->decode(plain_result, pod_result[i]);
    }
    u64** x_r = new u64*[a_dim];
    for(int i = 0; i < a_dim; i ++){
        x_r[i] = new u64[hidden_dim];
    }
    for(int i = 0; i < a_dim; i ++){
        for(int j = 0; j < hidden_dim; j++){
            x_r[i][j] = pod_result[i][j];
        }
    }

    return x_r;
}

SerialCT client_aggregation(const ServerFHE* sfhe, uint64_t a_dim, uint64_t hidden_dim, const uint64_t* client_vector, SerialCT ct, const uint64_t* const* a_matrix){
    auto context = static_cast<SEALContext*>(sfhe->context);
    // Recast needed fhe helpers and ciphertexts
    Encryptor *encryptor = reinterpret_cast<Encryptor*>(sfhe->encryptor);
    Evaluator *evaluator = reinterpret_cast<Evaluator*>(sfhe->evaluator);
    BatchEncoder *encoder = reinterpret_cast<BatchEncoder*>(sfhe->encoder);
    // 将client端的随机向量encode为Plaintext
    uv64 pod_matrix(8192, 0ULL);
    for(int row = 0; row < hidden_dim; row++){
        pod_matrix[row] = client_vector[row];
        // cout << "[" << row << "] "<< client_vector[row] << ",##############" << endl;
    }
    Plaintext enc_vec;
    encoder->encode(pod_matrix, enc_vec);

    // 将ct还原为a_dim个密文
    vector<Ciphertext> x_ct(a_dim);
    recast_opaque(ct, x_ct, context);

    // 将client端在relu层添加的随机向量消掉
    for(int i = 0; i < a_dim; i++){
        evaluator->sub_plain_inplace(x_ct[i],enc_vec);
    }

    // Define a zero Ciphertext
    uv64 pod_matrix_2(8192, 0ULL);
    Plaintext tmp;
    Ciphertext zero;
    encoder->encode(pod_matrix_2, tmp);
    encryptor->encrypt(tmp, zero);
    // 用来保存结果的密文
    vector<Ciphertext> encrypted_result(a_dim,zero);

    for(int i = 0; i < a_dim; i++){
        for(int j = 0; j < a_dim; j++){
            // if(a_matrix[i][j] == 1){
            if(a_matrix[i][j] > 0){
                // cout << "[" << i << "," << j << "] "<< a_matrix[i][j] << ",##############" << endl;
                evaluator->add_inplace(encrypted_result[i],x_ct[j]);
            }
        }
    }
    // for(int i = 0; i < a_dim; i++){
    //     delete a_matrix[i];
    // }
    // delete[] a_matrix;

    // 给计算结果添加上client端的随机向量
    for(int i = 0; i < a_dim; i++){
        evaluator->add_plain_inplace(encrypted_result[i],enc_vec);
    }

    // Serialize ciphertexts
    SerialCT serial_ct = serialize_ct(encrypted_result);
    return serial_ct;
}

void server_free_ax(uint64_t** ax, uint64_t a_dim){
    for(uint64_t i = 0; i < a_dim; i++){
        delete[] ax[i];
    }
    delete[] ax;
}


void knn()
{
    EncryptionParameters parms(scheme_type::bfv);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));

    /*
    To enable batching, we need to set the plain_modulus to be a prime number
    congruent to 1 modulo 2*poly_modulus_degree. Microsoft SEAL provides a helper
    method for finding such a prime. In this example we create a 20-bit prime
    that supports batching.
    */
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));

    SEALContext context(parms);

    /*
    We can verify that batching is indeed enabled by looking at the encryption
    parameter qualifiers created by SEALContext.
    */
    auto qualifiers = context.first_context_data()->qualifiers();
    cout << "Batching enabled: " << boolalpha << qualifiers.using_batching << endl;

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    /*
    Batching is done through an instance of the BatchEncoder class.
    */
    BatchEncoder batch_encoder(context);
    
    size_t slot_count = batch_encoder.slot_count();
    size_t row_size = slot_count / 2;

    clock_t start = clock();
    // Difine matrix X
    size_t a_dim = 1021; 
    vector<uint64_t> x_row_1(slot_count, 0ULL);
    vector<vector<uint64_t>> x_row(a_dim,x_row_1);

    /*
    Encrypt the Matrix X
    */

    vector<Plaintext> plain_x_row(a_dim);
    vector<Ciphertext> encrypted_x_row(a_dim);
    for (size_t i = 0; i < a_dim; i++){
        batch_encoder.encode(x_row[i],plain_x_row[i]);
        encryptor.encrypt(plain_x_row[i], encrypted_x_row[i]);
    }

    // Define the adjaeceny matrix
    
    int a[42][42] = {
        {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
        {0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0},
        {0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1}};

    // Define a zero Ciphertext
    vector<uint64_t> pod_matrix(slot_count, 0ULL);
    Plaintext tmp;
    Ciphertext zero;
    batch_encoder.encode(pod_matrix, tmp);
    encryptor.encrypt(tmp, zero);

    // 使用add_inplace需要初始化
    // vector<Ciphertext> encrypted_result(a_dim,zero);

    // 使用add_many不需要初始化
    // 实际使用过程中不能使用add_may,因为可能存在邻接矩阵A的一行全是0的情况,这种情况下，结果矩阵的该行将没有结果
    vector<Ciphertext> encrypted_result(a_dim);

    // Compute the Matrix Multiplication
    // for (size_t i = 0; i < a_dim; i++){
    //     vector<Ciphertext> result_ct;
    //     for (size_t j = 0; j < a_dim; j++){
    //         if(a[i][j]==1){
    //             result_ct.push_back(encrypted_x_row[j]);
    //             // evaluator.add_inplace(encrypted_result[i],encrypted_x_row[j]);
    //         }
    //     }
    //     evaluator.add_many(result_ct,encrypted_result[i]);
    // }
    for (size_t i = 0; i < a_dim; i++){
        vector<Ciphertext> result_ct;
        for (size_t j = 0; j < a_dim; j++){
            if(j<3){
                cout << i << "," << j << endl;
                result_ct.push_back(encrypted_x_row[j]);
                // evaluator.add_inplace(encrypted_result[i],encrypted_x_row[j]);
            }
        }
        evaluator.add_many(result_ct,encrypted_result[i]);
    }
    
    vector<Plaintext> plain_result(a_dim);
    vector<vector<uint64_t>> pod_result(a_dim);
    
    for (size_t i = 0; i < a_dim; i++){
        decryptor.decrypt(encrypted_result[i], plain_result[i]);
        batch_encoder.decode(plain_result[i], pod_result[i]);
    }
    clock_t end = clock();
    cout << double(end - start) / CLOCKS_PER_SEC << "s " <<  endl; // 打印0.001s

}