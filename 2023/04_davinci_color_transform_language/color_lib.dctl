// This code is a DCTL version converted from "https://github.com/ampas/aces-dev".
// Therefore, it follows the License Terms for Academy Color Encoding System Components.

// # License Terms for Academy Color Encoding System Components #
// 
// Academy Color Encoding System (ACES) software and tools are provided by the
// Academy under the following terms and conditions: A worldwide, royalty-free,
// non-exclusive right to copy, modify, create derivatives, and use, in source and
// binary forms, is hereby granted, subject to acceptance of this license.
// 
// Copyright © 2015 Academy of Motion Picture Arts and Sciences (A.M.P.A.S.).
// Portions contributed by others as indicated. All rights reserved.
// 
// Performance of any of the aforementioned acts indicates acceptance to be bound
// by the following terms and conditions:
// 
// * Copies of source code, in whole or in part, must retain the above copyright
// notice, this list of conditions and the Disclaimer of Warranty.
// 
// * Use in binary form must retain the above copyright notice, this list of
// conditions and the Disclaimer of Warranty in the documentation and/or other
// materials provided with the distribution.
// 
// * Nothing in this license shall be deemed to grant any rights to trademarks,
// copyrights, patents, trade secrets or any other intellectual property of
// A.M.P.A.S. or any contributors, except as expressly stated herein.
// 
// * Neither the name "A.M.P.A.S." nor the name of any other contributors to this
// software may be used to endorse or promote products derivative of or based on
// this software without express prior written permission of A.M.P.A.S. or the
// contributors, as appropriate.
// 
// This license shall be construed pursuant to the laws of the State of
// California, and any disputes related thereto shall be subject to the
// jurisdiction of the courts therein.
// 
// Disclaimer of Warranty: THIS SOFTWARE IS PROVIDED BY A.M.P.A.S. AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
// NON-INFRINGEMENT ARE DISCLAIMED. IN NO EVENT SHALL A.M.P.A.S., OR ANY
// CONTRIBUTORS OR DISTRIBUTORS, BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, RESITUTIONARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// WITHOUT LIMITING THE GENERALITY OF THE FOREGOING, THE ACADEMY SPECIFICALLY
// DISCLAIMS ANY REPRESENTATIONS OR WARRANTIES WHATSOEVER RELATED TO PATENT OR
// OTHER INTELLECTUAL PROPERTY RIGHTS IN THE ACADEMY COLOR ENCODING SYSTEM, OR
// APPLICATIONS THEREOF, HELD BY PARTIES OTHER THAN A.M.P.A.S.,WHETHER DISCLOSED OR
// UNDISCLOSED.


// Base functions from SMPTE ST 2084-2014

// Constants from SMPTE ST 2084-2014
const float pq_m1 = 0.1593017578125; // ( 2610.0 / 4096.0 ) / 4.0;
const float pq_m2 = 78.84375; // ( 2523.0 / 4096.0 ) * 128.0;
const float pq_c1 = 0.8359375; // 3424.0 / 4096.0 or pq_c3 - pq_c2 + 1.0;
const float pq_c2 = 18.8515625; // ( 2413.0 / 4096.0 ) * 32.0;
const float pq_c3 = 18.6875; // ( 2392.0 / 4096.0 ) * 32.0;

const float pq_C = 10000.0;
const float pq_Linear = 100.0;

// Converts from the non-linear perceptually quantized space to linear cd/m^2
// Note that this is in float, and assumes normalization from 0 - 1
// (0 - pq_C for linear) and does not handle the integer coding in the Annex 
// sections of SMPTE ST 2084-2014
__DEVICE__ float ST2084_2_Y( float N )
{
  // Note that this does NOT handle any of the signal range
  // considerations from 2084 - this assumes full range (0 - 1)
  float Np = _powf( N, 1.0 / pq_m2 );
  float L = Np - pq_c1;
  if( L < 0.0 ){
    L = 0.0;
  }
  L = L / ( pq_c2 - pq_c3 * Np );
  L = _powf( L, 1.0 / pq_m1 );
  return L * pq_C; // returns cd/m^2
}

// Converts from the non-linear perceptually quantized space to linear
// Note that this is in float, and assumes normalization from 0 - 1
// (0 - pq_Linear for linear) and does not handle the integer coding in the Annex 
// sections of SMPTE ST 2084-2014
__DEVICE__ float ST2084_2_Linear( float N )
{
  // Note that this does NOT handle any of the signal range
  // considerations from 2084 - this assumes full range (0 - 1)
  float Np = _powf( N, 1.0 / pq_m2 );
  float L = Np - pq_c1;
  if( L < 0.0 ){
    L = 0.0;
  }
  L = L / ( pq_c2 - pq_c3 * Np );
  L = _powf( L, 1.0 / pq_m1 );
  return L * pq_Linear; // returns linear value (1.0 means 100 cd/m2)
}

// Converts from linear cd/m^2 to the non-linear perceptually quantized space
// Note that this is in float, and assumes normalization from 0 - 1
// (0 - pq_C for linear) and does not handle the integer coding in the Annex 
// sections of SMPTE ST 2084-2014
__DEVICE__ float Y_2_ST2084( float C )
//pq_r
{
  // Note that this does NOT handle any of the signal range
  // considerations from 2084 - this returns full range (0 - 1)
  float L = C / pq_C;
  float Lm = _powf( L, pq_m1 );
  float N = ( pq_c1 + pq_c2 * Lm ) / ( 1.0 + pq_c3 * Lm );
  N = _powf( N, pq_m2 );
  return N;
}

// Converts from linear to the non-linear perceptually quantized space
// Note that this is in float, and assumes normalization from 0 - 1
// (0 - pq_Linear for linear) and does not handle the integer coding in the Annex 
// sections of SMPTE ST 2084-2014
__DEVICE__ float Linear_2_ST2084( float C )
//pq_r
{
  // Note that this does NOT handle any of the signal range
  // considerations from 2084 - this returns full range (0 - 1)
  float L = C / pq_Linear;
  float Lm = _powf( L, pq_m1 );
  float N = ( pq_c1 + pq_c2 * Lm ) / ( 1.0 + pq_c3 * Lm );
  N = _powf( N, pq_m2 );
  return N;
}

__DEVICE__ float3 Y_2_ST2084_f3( float3 in )
{
  // converts from linear cd/m^2 to PQ code values
  
  float3 out;
  out.x = Y_2_ST2084( in.x );
  out.y = Y_2_ST2084( in.y );
  out.z = Y_2_ST2084( in.z );

  return out;
}

__DEVICE__ float3 ST2084_2_Y_f3( float3 in )
{
  // converts from PQ code values to linear cd/m^2
  
  float3 out;
  out.x = ST2084_2_Y( in.x );
  out.y = ST2084_2_Y( in.y );
  out.z = ST2084_2_Y( in.z );

  return out;
}

__DEVICE__ float3 Linear_2_ST2084_f3( float3 in )
{
  // converts from linear to PQ code values
  
  float3 out;
  out.x = Linear_2_ST2084( in.x );
  out.y = Linear_2_ST2084( in.y );
  out.z = Linear_2_ST2084( in.z );

  return out;
}

__DEVICE__ float3 ST2084_2_Linear_f3( float3 in )
{
  // converts from PQ code values to linear
  
  float3 out;
  out.x = ST2084_2_Linear( in.x );
  out.y = ST2084_2_Linear( in.y );
  out.z = ST2084_2_Linear( in.z );

  return out;
}
