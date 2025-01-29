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
typedef unsigned long      Count;
typedef int                Boolean;
typedef long long          Word;

typedef struct {
	Bits64 *pointer;
	Index   index;
} BitLocation;

#define LMASK(x) (~(-1ll << (x)))

static BitLocation FindClearBit(Bits64 *p, Bits64 *q) {
	BitLocation result;
	__mmask8 m;
	__m512i v, cmp;
	
	cmp = _mm512_set1_epi64(-1);
	for (;;) {
		if (p >= q) break;
		v = _mm512_load_epi64(p);
		m = _mm512_cmplt_epu64_mask(v, cmp);
		if (m) break;
		p += sizeof(v) / sizeof(*p);
	}

	int tzcnt = _tzcnt_u32(m);
	result.pointer = p + tzcnt;
	if (result.pointer < q)
		(void)_BitScanForward64(&result.index, ~*result.pointer);
	return result;
}

#define WIDTHOF(x) (sizeof(x) * 8)
#define _ffsll(x)  __builtin_ffsll(x)

BitLocation FindClearBits(Size n, Bits64 *p, Bits64 *q) {
	if (n == 1) return FindClearBit(p, q);
	
	BitLocation result;
	Size c, i, j, z;
	Bits64 w;
	
	result.pointer = p;
	result.index = 0;
	c = 0;
	for (; p < q; ++p) {
		w = ~*p;
	retry:
		result.pointer = p;
		i = _ffsll(w);
		if (!i) continue;
		result.index = i - 1;
		w |= ~(-1ll << (i - 1));
		j = _ffsll(~w);
		if (!j) j = WIDTHOF(*p) + 1;
		c += j - i;
		if (c >= n) break;
		w &= -1ll << (j - 1);
		if (!w) continue;
		if (j == WIDTHOF(*p) + 1) {
			do {
				w = *++p;
				if (p >= q) goto finished;
				z = _tzcnt_u64(w);
				c += z;
				if (c >= n) goto finished;
			} while (z == WIDTHOF(*p));
			w |= ~(-1ll << (z - 1));
			w = ~w;
		}
		c = 0;
		goto retry;
	}
finished:
	if (p >= q) result = (BitLocation){0, 0};
	return result;
}

BitLocation correct;

#if 0
#define WIDTHOF(x) (sizeof(x) * 8)

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
#endif

size_t ClockRate;
size_t Start, End;

#define BeginClock() (void)QueryPerformanceCounter(&Start)
#define EndClock()   (void)QueryPerformanceCounter(&End)
#define Elapse()     ((End - Start) * 1000000000 / ClockRate)

#define N 10000
_Alignas(__m512i) Bits64 p[N];
Bits64 *q = p + N;

int main(int argc, char *argv[]) {
	QueryPerformanceFrequency(&ClockRate);
	memset(p, -1, N * sizeof(Bits64));
	unsigned long long random = 1000;
	_rdrand64_step(&random);
	correct.pointer = p + random % N;
	memset(correct.pointer, random, sizeof(Bits64));
	Assert(_BitScanForward64(&correct.index, ~*correct.pointer));
	printf("random %% N: %llu\n", random % N);
	printf("random: %llu\n", random);
	printf("correct.pointer: %p -> %llu\n", correct.pointer, *correct.pointer);
	printf("correct.index: %lu\n", correct.index);
	printf("q: %p\n", q);

	BitLocation result;

	QueryPerformanceCounter(&Start);
	result = FindClearBit(p, q);
	QueryPerformanceCounter(&End);
	printf("elapse: %llu\n", Elapse());
	printf("result: %p -> %llu & %lu\n", result.pointer, *result.pointer, result.index);
	
	QueryPerformanceCounter(&Start);
	result = FindClearBits(3, p, q);
	QueryPerformanceCounter(&End);
	printf("elapse: %llu\n", Elapse());
	printf("result: %p -> %llu & %lu\n", result.pointer, result.pointer ? *result.pointer : 0, result.index);
}
