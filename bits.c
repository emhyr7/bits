#include <stdio.h>
#include <immintrin.h>
#include <intrin.h>
#include <stdlib.h>
#include <memory.h>

extern int __stdcall QueryPerformanceCounter  (unsigned long long *);
extern int __stdcall QueryPerformanceFrequency(unsigned long long *);

extern void *__stdcall VirtualAlloc(void *, unsigned long long, unsigned, unsigned);

#define Assert(x) do { if (!(x)) __debugbreak(); } while (0)

typedef long long          Bits64;
typedef unsigned long long Size;
typedef unsigned long      Index;

typedef struct {
	Bits64 *pointer;
	Index   index;
} BitLocation;

#define LMASK(x) (~(-1ll << (x)))

static BitLocation FindClearBit2(Bits64 *p, Bits64 *q) {
	BitLocation result;
	__mmask8 m;
	__m512i v, cmp;

	cmp = _mm512_set1_epi64(-1);
	while (p < q) {
		v = _mm512_load_epi64(p);
		m = _mm512_cmplt_epu64_mask(v, cmp);
		if (m) break;
		p += 8;
	}

	int tzcnt = _tzcnt_u32(m);
	result.pointer = p + tzcnt;
	if (result.pointer < q) (void)_BitScanForward64(&result.index, ~*result.pointer);
	return result;
}

BitLocation correct;

/* don't use */
static BitLocation FindClearBits2(int n, Bits64 *p, Bits64 *q) {
	BitLocation result;
	unsigned long i;
	__mmask8 mask;
	__m512i v, vl, vh, vmask, vcmp, vlzcnt, vomask;
	
	Assert(n <= 32);

	vomask = _mm512_set1_epi64(-1);
	vmask = _mm512_set1_epi64(n);
	while (p < q) {
		v = _mm512_load_epi64(p);
		for (;;) {
			mask = _mm512_cmpeq_epi64_mask(v, vomask);
			if (mask == 0xff) break;
			vlzcnt = _mm512_lzcnt_epi64(v);
			mask = _mm512_cmpge_epi64_mask(vlzcnt, vmask);
			if (mask) goto finish;
			v = _mm512_sllv_epi64(v, vlzcnt);
			v = _mm512_xor_epi64(v, v);
			vlzcnt = _mm512_lzcnt_epi64(v);
			v = _mm512_sllv_epi64(v, vlzcnt);
			v = _mm512_xor_epi64(v, v);
		}
		mask = 0;

		p += 8;
	}
finish:
	int tzcnt = _tzcnt_u32(mask);
	result.pointer = p + tzcnt;
	result.index = 0;
	return result;
}

size_t ClockRate;
size_t Start, End;

#define BeginClock() (void)QueryPerformanceCounter(&Start)
#define EndClock()   (void)QueryPerformanceCounter(&End)
#define Elapse()     ((End - Start) * 1000000000 / ClockRate)

#define M 10000
#define N (sizeof(__m512i) / sizeof(Bits64) * (M))
_Alignas(__m512i) Bits64 p[N];
Bits64 *q = p + N;

int main(void) {
	QueryPerformanceFrequency(&ClockRate);
	memset(p, -1, N);
	unsigned long long random;
	_rdrand64_step(&random);
	correct.pointer = p + random % M;
	memset(correct.pointer, random, sizeof(__m512i));
	Assert(_BitScanForward64(&correct.index, ~*correct.pointer));
	printf("random %% M: %llu\n", random % M);
	printf("random: %llu\n", random);
	printf("correct.pointer: %p -> %llu\n", correct.pointer, *correct.pointer);
	printf("correct.index: %lu\n", correct.index);

	BitLocation result[2];

	QueryPerformanceCounter(&Start);
	result[0] = FindClearBit2(p, q);
	QueryPerformanceCounter(&End);
	printf("Elapse: %llu\n", Elapse());
	Assert(result[0].index == correct.index);
	Assert(result[0].pointer == correct.pointer);

	QueryPerformanceCounter(&Start);
	result[1] = FindClearBits2(1, p, q);
	QueryPerformanceCounter(&End);
	printf("Elapse: %llu\n", Elapse());

	for (int i = 0; i < (sizeof(result) / sizeof(result[0])); ++i) {
		printf("[%i]: %p & %lu\n", i, result[i].pointer, result[i].index);
	}
}
